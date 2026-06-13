#!/usr/bin/env python3
"""Game server for our GPU-native Liquid War clone — play + spectate.

The Python GPU engine (simulator/engine.py) is the single source of truth: it is
BOTH the RL training environment and the playable game. This serves one live
game over a websocket so any client (Canvas now, richer later) renders streamed
state and sends cursor input. No fidelity gap, no C-engine bridge — you play the
exact engine the policy trained in.

Smoothness: the server runs its OWN game loop (steps at ``LW_TICK_HZ`` and pushes
a frame each tick); the client's mouse input arrives async. So the frame rate is
steady and decoupled from the request/response round-trip (the old protocol's
jitter is gone).

Modes (websocket query param ``mode``):
  - ``play``     : you drive team 0 (mouse target -> cursor), the AI drives the
                   rest. Default ``teams=2`` -> a 1v1 duel (not a 3-on-1
                   pile-up). Opponent = ``opponent`` query param
                   ('heuristic' | 'latest' | a checkpoint path).
  - ``spectate`` : all teams AI — also how progress videos get recorded.

Run: ``uvicorn web.server:app --host 0.0.0.0 --port 8080``.
"""
from __future__ import annotations

import asyncio
import base64
import glob
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rl.eval import _heuristic_dydx
from rl.policy import LEGACY_ACTION, NUM_STANCES, CursorPolicy, act, apply_stances
from simulator.engine import LiquidWarEngine, GRADIENT_INF, MAP_NAMES

CKPT_DIR = os.environ.get("LW_CKPT_DIR", "/opt/training/results")  # NFS mount
DEVICE = os.environ.get("LW_PLAY_DEVICE", "cpu")
TICK_HZ = float(os.environ.get("LW_TICK_HZ", "60"))               # game + frame rate (60fps;
#   the deploy over-targets 63 so asyncio's ~1ms sleep granularity lands on a true 60)
_STATIC = Path(__file__).parent / "static"

# The tick loop is LAUNCH-BOUND: what kills fps is the launch thread losing
# the CPU to co-tenant batch jobs (measured: host load 48 -> 6fps). Keep the
# intraop pool tiny (frame_blob's CPU tensor ops are trivial) and raise our
# scheduling priority (works under --cap-add=SYS_NICE; harmless without).
torch.set_num_threads(2)
try:
    os.nice(-10)
except OSError:
    pass


def _latest_checkpoint() -> str | None:
    """Newest .pt under any run dir in CKPT_DIR (so the opponent tracks training)."""
    cands = glob.glob(os.path.join(CKPT_DIR, "rl", "*", "*.pt"))
    cands += glob.glob(os.path.join(CKPT_DIR, "*", "*.pt"))
    return max(cands, key=os.path.getmtime) if cands else None


def _load_policy(path: str) -> tuple[CursorPolicy, int]:
    """Load a checkpoint, sizing the stance head from the weights — legacy
    8-stance policies keep working (their actions are mapped through
    ``LEGACY_ACTION`` to the flat stance-mode space)."""
    sd = torch.load(path, map_location=DEVICE, weights_only=True)
    if "stance_head.2.weight" in sd:
        n_act = sd["stance_head.2.weight"].shape[0]
        policy = CursorPolicy(num_stances=n_act).to(DEVICE)
        policy.load_state_dict(sd)
    else:
        # PRE-STANCE era (move-only policy, June 2026-06-07 and earlier):
        # load the matching weights; the stance head stays untrained and the
        # session forces Classic — exactly how the game played in that era.
        n_act = 0
        policy = CursorPolicy().to(DEVICE)
        own = policy.state_dict()
        own.update({k: v for k, v in sd.items() if k in own and v.shape == own[k].shape})
        policy.load_state_dict(own)
    policy.eval()
    return policy, n_act


# PRACTICE MODE lessons: the dummy (all AI seats) holds a fixed stance each
# round so a new player learns to FACE each threat and which counter answers
# it. Cycles on each round; no win/loss/streak/gauntlet is recorded.
#   (flat action the dummy holds, coach line shown to the player)
LESSONS = [
    (24, "Free practice \u2014 hold keys 1\u20139 to try every stance"),
    (17, "Enemy holds \ud83d\udd73 DOOM \u2014 hold \ud83c\udf2a Maelstrom to shield, or kite away"),
    (8,  "Enemy holds \ud83d\udee1 WALL \u2014 \u27a4 Drill punches through it"),
    (20, "Enemy holds \ud83c\udf2a MAELSTROM \u2014 don't charge in; flank wide"),
    (6,  "Enemy \u27a4 DRILLS \u2014 \ud83d\udee1 Wall braces it"),
    (10, "Enemy \ud83d\udca5 PULSES \u2014 spread out so the crests miss"),
    (0,  "Enemy spreads a \ud83d\udd78 WEB \u2014 a dense Drill or Doom cuts through"),
]


class GameSession:
    """One live game: engine + a controller per team."""

    def __init__(self, mode: str, opponent: str, teams: int = 2,
                 height: int = 80, width: int = 110, fighters: int = 700) -> None:
        self.mode = mode
        self.engine = LiquidWarEngine(batch_size=1, height=height, width=width,
                                      num_teams=teams, fighters_per_team=fighters,
                                      device=DEVICE, grad_iters=24)
        # Cap gradient sweeps/tick so a fresh game's cold flood spreads over a few
        # frames (the persistent field accumulates) instead of one ~88ms stutter.
        self.engine._grad_cap = 48
        self.engine._spin = torch.ones(1, teams, device=DEVICE)  # per-team orbit sign; player flips team 0
        self.engine._burst = torch.zeros(1, teams, device=DEVICE)  # gather(-1)/burst(+1) phase; player drives 0
        self.engine._drill = torch.zeros(1, teams, 2, device=DEVICE)  # per-team thrust dir (drill move)
        self.engine._wall = torch.zeros(1, teams, 2, device=DEVICE)  # per-team shield facing (wall stance)
        self.engine._fig8 = torch.zeros(1, teams, device=DEVICE)  # per-team figure-8 flag (Atom stance)
        self.engine._ring = torch.zeros(1, teams, device=DEVICE)  # per-team orbit radius (Doom's accretion disk)
        self.engine._ring_ecc = torch.zeros(1, teams, device=DEVICE)  # ring shape: 1 oblate Gargantua (Doom) / 0 circular (Maelstrom)
        self.engine._node_l = torch.zeros(1, teams, device=DEVICE)  # Chladni radial wavelength (Pulse rings mode)
        self.engine._node_m = torch.zeros(1, teams, device=DEVICE)  # Chladni angular petals (Pulse star / Spin modes)
        self.engine._node_k = torch.zeros(1, teams, device=DEVICE)  # angular-mode radial pitch (galaxy spiral arms)
        self.engine._node_w = torch.zeros(1, teams, device=DEVICE)  # angular-mode rotation speed (sawblade sweep)
        self.engine._node_v = torch.zeros(1, teams, device=DEVICE)  # ring breathe speed (Pulse rings/star energy)
        self.engine._tide = torch.zeros(1, teams, 2, device=DEVICE)  # directional traveling-wave heading (Pulse tide mode)
        self.engine._surge = torch.ones(1, teams, device=DEVICE)  # per-team damage mult; IN-PLACE writes only (graph input)
        self.engine._armor = torch.ones(1, teams, device=DEVICE)  # per-team INCOMING-damage mult (Wall braces)
        self.engine._wells_enabled = True            # play casts real cross-team wells (slots in engine.reset)
        # CUDA graphs are OPT-IN for play now (LW_CUDA_GRAPH=1): with rooms
        # churning constantly (auto-reconnect, phones sleeping, multiple
        # devices) the capture/teardown lifecycle keeps re-poisoning the CUDA
        # context through a race we have not fully pinned — three narrowing
        # fixes (reset sync, teardown sync, eager small rooms) each helped but
        # big-room churn still hits it. Eager keeps ~most of the perf work
        # (sync-free hot path, pooled gradient, CPU cursors, policy cache);
        # stability beats the last few fps until the race is reproduced and
        # killed offline.
        self.engine._cuda_graph = os.environ.get("LW_CUDA_GRAPH", "0") == "1"
        self.engine.reset()
        self.policy: CursorPolicy | None = None
        self.practice = False              # forgiving sandbox (opponent='practice')
        self._lesson = 0                   # which LESSONS entry the dummy holds
        # GAME MODE (annihilate | koth). KOTH = a central hill; the team with
        # the clear majority of in-hill mass scores each tick, first to target
        # wins — a positional objective, not a fight to the last drop.
        self.gamemode = "annihilate"
        self.koth_target = int(os.environ.get("LW_KOTH_TARGET", "600"))
        self.koth_score = [0] * teams
        _hy, _hx = self.engine.H // 2, self.engine.W // 2
        _hr = max(8, min(self.engine.H, self.engine.W) // 5)
        self.hill = (_hy, _hx, _hr)
        _ys = torch.arange(self.engine.H, device=DEVICE).view(-1, 1)
        _xs = torch.arange(self.engine.W, device=DEVICE).view(1, -1)
        self._hill_mask = (((_ys - _hy) ** 2 + (_xs - _hx) ** 2) <= _hr * _hr).float()
        self._opp_request = opponent
        self._set_opponent(opponent)

    def _set_opponent(self, opponent: str) -> None:
        """Load (or re-roll) the AI. ``latest`` ROTATES per match: mostly the
        live champion, sometimes an archived generation — three eras play
        three genuinely different games, and one collapsed checkpoint can't
        define 'the AI' for every round (the perma-Doom complaint)."""
        self.policy = None
        self._legacy = None
        self.ckpt_name = opponent
        self.practice = (opponent == "practice")
        if opponent in ("heuristic", "random", "practice"):
            return
        if opponent == "latest":
            arch = sorted(glob.glob(os.path.join(CKPT_DIR, "rl", "archive", "*.pt")))
            path = (random.choice(arch) if arch and random.random() < 0.35
                    else _latest_checkpoint())
        else:                                  # roster pick: a path RELATIVE to CKPT_DIR only
            cand = (Path(CKPT_DIR) / opponent).resolve()
            path = (str(cand) if str(cand).startswith(str(Path(CKPT_DIR).resolve()) + os.sep)
                    and cand.is_file() else _latest_checkpoint())
        if path:
            self.policy, n_act = _load_policy(path)
            self.ckpt_name = Path(path).name
            # legacy small-head checkpoint (5- or 8-stance eras) -> the
            # flat actions those base stances meant; pre-stance era -> all
            # Classic (the pure blob); full-width heads map raw
            if n_act == 0:
                self._legacy = torch.full((NUM_STANCES,), NUM_STANCES - 1,
                                          dtype=torch.long, device=DEVICE)
            elif n_act <= len(LEGACY_ACTION):
                self._legacy = torch.tensor(LEGACY_ACTION[:n_act], device=DEVICE)
            else:
                self._legacy = None

    @property
    def done(self) -> bool:
        return bool(self.engine.team_alive[0].sum() <= 1)

    @property
    def koth_done(self) -> bool:
        return self.gamemode == "koth" and max(self.koth_score, default=0) >= self.koth_target

    @property
    def match_over(self) -> bool:
        return self.done or self.koth_done

    def winner(self) -> int:
        """Match winner: KOTH leader if the hill was won, else the largest army
        (covers annihilation AND a koth game that ends by elimination/timeout)."""
        if self.koth_done:
            return max(range(len(self.koth_score)), key=lambda t: self.koth_score[t])
        counts = self.engine.team_oh[0].sum(dim=(1, 2)).tolist()
        return max(range(len(counts)), key=lambda t: counts[t])

    def _score_koth(self) -> None:
        """+1 to the team holding a clear (>55%) majority of the hill's mass."""
        oh = self.engine.team_oh[0]                                   # (T,H,W)
        pres = (oh * self._hill_mask.unsqueeze(0)).sum(dim=(1, 2)).tolist()
        total = sum(pres)
        if total < 1.0:
            return
        lead = max(range(len(pres)), key=lambda t: pres[t])
        if pres[lead] > 0.55 * total and bool(self.engine.team_alive[0, lead]):
            self.koth_score[lead] += 1

    @torch.no_grad()
    def _ai_dydx(self):
        """``(dydx, ai_stance)`` for ALL teams from the configured AI. ``ai_stance``
        is the policy's chosen stance per team (``None`` for the heuristic, which
        has no stances).

        Two policy-agnostic guards sit between the network and the game (both
        born of the perma-Doom-3x checkpoint, but kept as safety nets):
          - DOOM GOVERNOR: a seat may hold Doom (flat 16-18) for at most
            LW_AI_DOOM_BUDGET consecutive ticks (~600 = 10s), then those
            logits are masked -inf for LW_AI_DOOM_COOLDOWN ticks (~360) and
            argmax falls to its best non-Doom plan. 0 disables.
          - STANCE MIXTURE: LW_AI_STANCE_TEMP > 0 samples the stance head at
            that temperature once per ~45 ticks and HOLDS it (training's
            K_HOLD cadence) — the AI shows its real mixture instead of the
            argmax mode. Default 0 (argmax) until a non-collapsed head ships.
        """
        if self.policy is None:
            if self.practice:                  # dummy holds this round's lesson stance
                act_id = LESSONS[self._lesson % len(LESSONS)][0]
                stance = torch.full((1, self.engine.T), act_id, dtype=torch.long, device=DEVICE)
                return _heuristic_dydx(self.engine), stance
            return _heuristic_dydx(self.engine), None
        e = self.engine
        T = e.T
        budget = int(os.environ.get("LW_AI_DOOM_BUDGET", "600"))
        cooldown = int(os.environ.get("LW_AI_DOOM_COOLDOWN", "360"))
        temp = float(os.environ.get("LW_AI_STANCE_TEMP", "0"))
        if getattr(self, "_doom_cool", None) is None or len(self._doom_cool) != T:
            self._doom_run = [0] * T
            self._doom_cool = [0] * T
            self._ai_hold = None
            self._ai_hold_t = -999
        cool, run = self._doom_cool, self._doom_run
        full_head = getattr(self, "_legacy", None) is None
        mask = None
        if budget > 0 and full_head and any(c > 0 for c in cool):
            mask = torch.zeros(1, T, NUM_STANCES, device=DEVICE)
            for t in range(T):
                if cool[t] > 0:
                    mask[0, t, 16:19] = float("-inf")
        n = int(e.tick)
        redecide = (temp <= 0 or self._ai_hold is None
                    or self._ai_hold.shape[1] != T or n - self._ai_hold_t >= 45)
        obs = e.get_observation()
        dydx, stance, _, _, _ = act(self.policy, obs, T, e.team_alive,
                                    deterministic=True,
                                    stance_temp=(temp if redecide else 0.0),
                                    stance_mask=mask)
        if temp > 0:
            if redecide:
                self._ai_hold = stance.clone()
                self._ai_hold_t = n
            else:
                stance = self._ai_hold
        # ANTI-CAMP ESCORT: the small-map lineage pins its cursor into the
        # play board's corner (measured 93% corner occupancy at 384x576 —
        # far out of training distribution, the move argmax degenerates to a
        # constant heading). If an AI seat sits in a corner box for more than
        # LW_AI_CAMP_BUDGET ticks, the heuristic coach steers that seat for
        # LW_AI_CAMP_ESCORT ticks (toward the fight, like a player would),
        # then the policy gets the wheel back. 0 disables.
        camp_budget = int(os.environ.get("LW_AI_CAMP_BUDGET", "120"))
        escort_len = int(os.environ.get("LW_AI_CAMP_ESCORT", "900"))
        if camp_budget > 0:
            if getattr(self, "_camp_run", None) is None or len(self._camp_run) != T:
                self._camp_run = [0] * T
                self._camp_escort = [0] * T
            cur = e.cursor_pos[0].tolist()
            hdydx = None
            for t in range(T):
                cy, cx = cur[t]
                in_corner = (min(cy, e.H - 1 - cy) < 0.12 * e.H
                             and min(cx, e.W - 1 - cx) < 0.12 * e.W)
                if self._camp_escort[t] > 0:                  # coach has the wheel
                    self._camp_escort[t] -= 2                 # (inference every 2nd tick)
                    if hdydx is None:
                        hdydx = _heuristic_dydx(e)
                    dydx[0, t] = hdydx[0, t]
                elif in_corner:
                    self._camp_run[t] += 2
                    if self._camp_run[t] >= camp_budget:
                        self._camp_run[t] = 0
                        self._camp_escort[t] = escort_len
                else:
                    self._camp_run[t] = 0
        if not full_head:                                 # old 5/8-stance policy.pt
            stance = self._legacy[stance]
            if budget > 0 and any(c > 0 for c in cool):   # no logits to mask: override
                for t in range(T):
                    if cool[t] > 0 and 16 <= int(stance[0, t]) <= 18:
                        stance[0, t] = NUM_STANCES - 1    # Classic during cooldown
        # governor bookkeeping (inference runs every 2nd tick -> count by 2)
        acts = stance[0].tolist()
        for t in range(T):
            if cool[t] > 0:
                cool[t] = max(0, cool[t] - 2)
                run[t] = 0
            elif 16 <= acts[t] <= 18:
                run[t] += 2
                if budget > 0 and run[t] >= budget:
                    cool[t] = cooldown
                    run[t] = 0
                    self._ai_hold = None                  # masked seat re-decides NOW
            else:
                run[t] = 0
        return dydx, stance

    @torch.no_grad()
    def _mouse_dir(self, t: int, target) -> list[int]:
        """Wall-aware mouse homing. The old greedy sign(target-cursor) pinned
        the cursor on any concave barrier; instead, flood a distance field
        from the CLICK (the same pooled octile relax as the army gradient,
        amortized ~64 sweeps/tick so a click costs a few transient ms) and
        step the cursor downhill — it routes around walls like a fighter.
        The field is static once converged (walls don't move), so a held
        target costs nothing at steady state."""
        e = self.engine
        ty = max(1, min(e.H - 2, int(target[0])))
        tx = max(1, min(e.W - 2, int(target[1])))
        clicks = getattr(self, "_clicks", None)
        if clicks is None:
            clicks = self._clicks = {}
        st = clicks.get(t)
        if st is None or st["tgt"] != (ty, tx):
            f = torch.full((1, 1, e.H, e.W), 1e9, device=e.walls.device)
            f[0, 0, ty, tx] = 0.0
            st = clicks[t] = {"tgt": (ty, tx), "f": f, "left": 2 * (e.H + e.W)}
        cy, cx = e.cursor_pos[0, t].tolist()
        if st["left"] > 0:
            mp = torch.nn.functional.max_pool2d
            wall = e.walls[0].unsqueeze(0).unsqueeze(0)
            f = st["f"]
            for _ in range(min(64, st["left"])):
                ng = -f
                orth = torch.maximum(mp(ng, (1, 3), stride=1, padding=(0, 1)),
                                     mp(ng, (3, 1), stride=1, padding=(1, 0)))
                diag = mp(ng, 3, stride=1, padding=1)
                f = torch.minimum(f, torch.minimum(10.0 - orth, 14.0 - diag))
                f = torch.where(wall, torch.full_like(f, 1e9), f)
            st["f"] = f
            st["left"] -= min(64, st["left"])
            # once the flood has reached the cursor, a little polish then stop
            if float(st["f"][0, 0, cy, cx]) < 1e8:
                st["left"] = min(st["left"], 128)
        fg = st["f"][0, 0]
        fval = float(fg[cy, cx])
        # ARRIVE, don't orbit: the cursor steps cursor_speed (~6) cells/tick,
        # so near the click it overshoots back and forth forever (the jitter).
        # Cap this seat's speed at the remaining PATH distance (the flood
        # field's own value /10; Chebyshev fallback pre-flood) — full speed on
        # detours, decelerating to land exactly on the clicked cell.
        rem = (int(fval // 10) + (1 if fval % 10 else 0)) if fval < 1e8 \
            else max(abs(ty - cy), abs(tx - cx))
        spd = getattr(e, "_cursor_speed_t", None)
        if spd is not None:
            spd[t] = max(0, min(spd[t], rem))
        if fval >= 1e8:                              # flood not here yet: greedy meanwhile
            return [max(-1, min(1, ty - cy)), max(-1, min(1, tx - cx))]
        patch = fg[cy - 1:cy + 2, cx - 1:cx + 2]     # cursor lives in 1..H-2, safe
        k = int(patch.argmin())
        return [k // 3 - 1, k % 3 - 1]

    @torch.no_grad()
    def step(self, humans: dict[int, tuple] | None = None) -> None:
        """Advance one tick. ``humans`` maps each occupied seat (team index)
        to its player's ``(mouse_target, key_dir)``; every other seat is
        AI-driven (knobs + wells via apply_stances' seat masking)."""
        humans = humans or {}
        # Policy inference every 2nd tick: the engine still steps at full rate,
        # the AI just holds its dydx/stance for ~32ms — imperceptible, and it
        # buys back ~half the inference cost (the last ms to a locked 60fps).
        cache = getattr(self, "_ai_cache", None)
        if cache is None or self.engine.tick % 2 == 0:
            raw_dydx, ai_stance = self._ai_dydx()
            self._ai_cache = (raw_dydx, ai_stance)
        else:
            raw_dydx, ai_stance = cache
        dydx = raw_dydx.clone()                   # human rows are written below
        if self.mode == "play":
            for t, (target, hdir) in humans.items():
                if hdir is not None and (hdir[0] or hdir[1]):
                    # Keyboard (arrows/WASD): drive the cursor directly, LW-style.
                    dydx[0, t, 0] = max(-1, min(1, hdir[0]))
                    dydx[0, t, 1] = max(-1, min(1, hdir[1]))
                elif target is not None:
                    # Mouse: wall-aware homing — flood-field downhill (see
                    # _mouse_dir), so clicks across a barrier route around it.
                    d = self._mouse_dir(t, target)
                    dydx[0, t, 0] = d[0]
                    dydx[0, t, 1] = d[1]
                else:
                    # No input: a human seat stays put — never let the AI
                    # drive a player's cursor.
                    dydx[0, t, 0] = 0
                    dydx[0, t, 1] = 0
        human_set = set(humans) if self.mode == "play" else set()
        self._ai_doom = None
        self._ai_mael = None
        if ai_stance is not None:                  # AI fills every seat without a human
            # (apply_stances also writes the AI seats' cursor speeds — the
            # Doom mobility tax lives in the knob table now, train AND play)
            apply_stances(self.engine, ai_stance, dydx, human_teams=human_set)
            self._ai_fx_markers(ai_stance, human_set)
        # overtime melts fronts for EVERY opponent type — it used to sit inside
        # the policy branch above, so heuristic/random games never concluded
        ot = getattr(self, "_overtime", 1.0)
        if ot > 1.0:
            self.engine._surge.mul_(ot)
        self.engine.step(dydx)
        if self.gamemode == "koth":
            self._score_koth()

    def _ai_fx_markers(self, ai_stance: torch.Tensor, human_set=()) -> None:
        """HUD markers for the client's shaders: the first AI team holding a
        Doom action (16-18, charge level encoded) feeds the black-hole shader,
        the first holding Maelstrom (19-21) feeds the whirlpool. The wells
        themselves are cast by ``apply_stances``."""
        e = self.engine
        acts = ai_stance[0].tolist()
        alive = e.team_alive[0].tolist()
        for t in range(e.T):
            a = acts[t]
            if not alive[t] or t in human_set:
                continue
            if 16 <= a <= 18 and self._ai_doom is None:
                self._ai_doom = [t, *e.cursor_pos[0, t].tolist(), a - 15]
            elif 19 <= a <= 21 and self._ai_mael is None:
                # [..., mode] — undertow/ejecta/shear render as different storms
                self._ai_mael = [t, *e.cursor_pos[0, t].tolist(), a - 19]

    def reset(self) -> None:
        if self._opp_request == "latest":      # per-match rotation (see _set_opponent)
            self._set_opponent("latest")
        self.koth_score = [0] * self.engine.T
        self.engine.reset()
        self._prev_pos = None                  # force a keyframe after every reset
        self._grid_burst = 8                   # first frames carry the grid (cold flood)
        self._last_blob = None                 # never serve the OLD map's frozen frame
        self._last_blob_tick = None

    def frame_blob(self) -> bytes:
        """The per-tick BINARY channel: mote positions + teams (+ the cell
        grid at ~5 Hz). Fighters move <= unit_speed (6) cells/tick, so between
        periodic int16 KEYFRAMES every position update is an int8 DELTA —
        about half the bytes of the old base64-in-JSON stream, with zero
        client-side atob/JSON cost. Layout (little-endian):
          u8 type (1=keyframe i16 abs, 2=delta i8) | u8 hasGrid |
          u16 pn | u32 seq | pos[2*pn] | pteam u8[pn] | grid i8[H*W]?
        Cadence runs on a SEND counter, not the engine tick: while the tick is
        frozen (3-2-1 countdown, the 20s result hold) the old tick-modulo test
        was stuck true and streamed a full keyframe + grid EVERY frame —
        ~14 MB/s per client of identical bytes on the big board. ``seq`` also
        lets the client detect a dropped frame and re-sync on the next
        keyframe instead of accumulating deltas across the gap (mote smear).
        """
        e = self.engine
        # FROZEN ENGINE, FREE FRAME: while the tick isn't advancing (3-2-1
        # countdown, the 20s result hold) the world is static — serve the
        # cached blob and skip the GPU->CPU transfers entirely. The client
        # drops repeated-seq deltas by design; the first live frame after a
        # freeze forces a keyframe so it re-syncs instantly.
        tick_now = int(e.tick)
        if tick_now == getattr(self, "_last_blob_tick", -1) \
                and getattr(self, "_last_blob", None) is not None:
            self._frozen = True
            return self._last_blob
        self._last_blob_tick = tick_now
        if getattr(self, "_frozen", False):
            self._frozen = False
            self._prev_pos = None
        self._fseq = getattr(self, "_fseq", 0) + 1
        step = max(1, e.N // 9000)
        pidx = torch.arange(0, e.N, step, device=e.device)
        pos = torch.stack((e.fy[0, pidx], e.fx[0, pidx]), dim=1).reshape(-1).to(torch.int16).cpu()
        prev = getattr(self, "_prev_pos", None)
        key = prev is None or prev.numel() != pos.numel() or self._fseq % 30 == 0
        self._prev_pos = pos
        body = (pos if key else (pos - prev).to(torch.int8)).numpy().tobytes()
        pteam = e.fteam[0, pidx].to(torch.uint8).cpu().numpy().tobytes()
        burst = getattr(self, "_grid_burst", 8)
        send_grid = (self._fseq % 12 == 0) or burst > 0
        if burst > 0:
            self._grid_burst = burst - 1
        grid = b""
        gflag = 0
        if send_grid:
            oh = e.team_oh[0]
            present = oh.sum(0) > 0
            cell = oh.argmax(0).to(torch.int8)
            cell = torch.where(present, cell, torch.full_like(cell, -1))
            cell = torch.where(e.walls[0], torch.full_like(cell, -2), cell)  # -2 = wall
            # RLE the cell grid: it's long homogeneous runs (walls / empty /
            # team blocks), and the doubled board made the raw grid ~440KB.
            # Wire: (i8 value, u16 runlen) triples; gflag=2. A battle grid
            # compresses heavily; on a pathologically fragmented frame RLE can
            # EXPAND, so fall back to raw (gflag=1) — never worse than before.
            raw = cell.cpu().numpy().reshape(-1)
            rle = _rle_grid(raw)
            if len(rle) < raw.size:
                grid = rle; gflag = 2
            else:
                grid = raw.tobytes(); gflag = 1
        import struct
        head = struct.pack("<BBHI", 1 if key else 2, gflag,
                           pidx.numel(), self._fseq & 0xFFFFFFFF)
        self._last_blob = head + body + pteam + grid
        return self._last_blob

    def state(self) -> dict[str, Any]:
        e = self.engine
        oh = e.team_oh[0]                                  # (T,H,W) presence
        # (motes + grid travel on the BINARY channel — see frame_blob)
        # Telemetry: gradient coverage (is the flood-fill complete?) and per-team
        # army spread = mean fighter distance to its own cursor. Spread should
        # FALL as the army flows in and packs around the cursor; a stuck/funneled
        # army keeps a high spread — the quantitative read on "does it flow."
        # HUD metrics (counts / flood / spread) each force a GPU->CPU sync; they're
        # display-only, so recompute at ~10Hz (every 6 ticks) and cache — keeps the
        # per-frame sync count (hence the frame rate) low. Render-essential fields
        # (grid, cursors, alive) stay per-frame.
        # gate on the tick ADVANCING: with the tick frozen at 0 through a
        # countdown, `% 6 == 0` re-ran this multi-sync block every iteration
        if not hasattr(self, "_hud") or (int(e.tick) % 6 == 0
                                         and int(e.tick) != getattr(self, "_last_hud_tick", -1)):
            self._last_hud_tick = int(e.tick)
            reach = int((~e.walls[0]).sum())
            flood = int((e.gradient[0, 0] < GRADIENT_INF).sum())
            spread = []
            for t in range(e.T):
                ys, xs = torch.where(oh[t] > 0)
                if ys.numel():
                    cy, cx = e.cursor_pos[0, t].float()
                    spread.append(round(float(((ys.float() - cy) ** 2 + (xs.float() - cx) ** 2).sqrt().mean()), 1))
                else:
                    spread.append(0.0)
            self._hud = {
                "fighters": oh.sum(dim=(1, 2)).long().tolist(),
                "flood_pct": round(100 * flood / max(reach, 1)),
                "spread": spread,
            }
        return {
            "tick": int(e.tick),
            "h": e.H, "w": e.W, "teams": e.T,
            "opponent": self.ckpt_name,
            "cursors": e.cursor_pos[0].tolist(),
            "alive": e.team_alive[0].tolist(),
            "ai_doom": getattr(self, "_ai_doom", None),   # [team, y, x, lvl] while an AI holds Doom
            "ai_mael": getattr(self, "_ai_mael", None),   # [team, y, x] while an AI holds Maelstrom (whirl shader)

            "done": self.done,
            "map": MAP_NAMES[e._last_arch] if 0 <= getattr(e, "_last_arch", -1) < len(MAP_NAMES) else "?",
            **self._hud,
        }


app = FastAPI(title="liquidwar play")


STANCES = ("Swarm", "Spin", "Drill", "Wall", "Pulse", "Doom", "Maelstrom", "Atom", "Classic")

PLAYER_CTRL = {"target": None, "dir": None, "reset": False,
               "map": None, "spin": None, "stance": 0, "drill_mode": 0, "doom_level": 1,
               "wall_orient": 0,   # 0 = horizontal bar, 1 = vertical bar (tap 4 to flip)
               "pulse_mode": 0,    # wave/rings/star/lattice/nova/tide (tap 5 to cycle)
               "spin_mode": 0,     # vortex/sawblade/galaxy (tap 2 to cycle)
               "mael_mode": 0,     # undertow/ejecta/shear (tap 7 to cycle)
               "swarm_mode": 0,    # cloud/comet (tap 1 to cycle)
               "atom_mode": 0}     # orbital/binary (tap 8 to cycle)


def _mode_name(ctrl) -> str:
    return (("slow", "med", "fast")[ctrl["drill_mode"]] if ctrl["stance"] == 2
            else f"{ctrl['doom_level']}x" if ctrl["stance"] == 5
            else ("horiz", "vert")[ctrl["wall_orient"]] if ctrl["stance"] == 3
            else ("wave", "rings", "star", "lattice", "nova", "tide")[ctrl["pulse_mode"]] if ctrl["stance"] == 4
            else ("vortex", "sawblade", "galaxy")[ctrl["spin_mode"]] if ctrl["stance"] == 1
            else ("undertow", "ejecta", "shear")[ctrl["mael_mode"]] if ctrl["stance"] == 6
            else ("web", "comet")[ctrl["swarm_mode"]] if ctrl["stance"] == 0
            else ("orbital", "binary")[ctrl["atom_mode"]] if ctrl["stance"] == 7 else "")


def _apply_player_stance(_e, t, ctrl, spin_sign, last_dir, c0_hist, n,
                         morph: int = 0) -> int:
    """Apply ONE human player's held stance to team ``t``'s knobs — the exact
    block that used to be inlined for team 0, parametrized by seat so any
    number of humans can hold stances in the same game. Returns the player's
    cursor speed (Doom charge trades speed for pull).

    ``morph`` > 0 = ticks remaining in the stance-SWITCH transient: the army
    flares open then SNAPS into the new form with an over-revved spin — the
    switch reads as a burst of energy instead of a soft fade."""
    stance = ctrl["stance"]                     # 0 Swarm 1 Spin 2 Drill 3 Wall 4 Pulse
    if stance == 0:                             # Swarm: 2 forms (tap 1): web -> comet
        if ctrl["swarm_mode"] == 0:             # web: a living NET of strands
            # concentric rings (node_l) crossed with straight radial spokes
            # (node_m, k=0 — NOT spiraled like Pulse-lattice) = a polar grid
            # = a spider web. The units TRAVEL through it: the node terms are
            # traveling waves, so a faster radial phase (node_v) migrates the
            # rings and units ride them IN and OUT, a faster angular sweep
            # (node_w) carries the spokes AROUND, and a healthy orbital spin
            # streams units ALONG the threads (bunching where they cross a
            # spoke) — a swarming current flowing through a living web.
            _e._spin[0, t] = 0.7 * spin_sign    # orbital traffic around the net
            _e._burst[0, t] = 0.25              # the ring radii define the spread
            _e._surge[0, t] = 1.2               # a thousand small bites
            _e._node_l[0, t] = 14.0             # concentric ring spacing
            _e._node_m[0, t] = 12.0             # 12 radial spokes -> a fine net
            _e._node_w[0, t] = 0.09 * (spin_sign if spin_sign != 0 else 1)  # spokes sweep AROUND
            _e._node_v[0, t] = 0.07             # rings travel IN and OUT (units ride them)
        else:                                   # comet: a teardrop along your MOTION
            sgn = spin_sign if spin_sign != 0 else 1
            # aim = recent cursor displacement (works for mouse AND
            # keys, unlike last_dir); standing still relaxes the
            # comet back into a blob
            cdy = (c0_hist[-1][0] > c0_hist[0][0]) - (c0_hist[-1][0] < c0_hist[0][0])
            cdx = (c0_hist[-1][1] > c0_hist[0][1]) - (c0_hist[-1][1] < c0_hist[0][1])
            _e._drill[0, t, 0] = 0.95 * cdy     # HEAVY head punches along the motion
            _e._drill[0, t, 1] = 0.95 * cdx     #   (the drill machinery, velocity-aimed)
            _e._spin[0, t] = 0.2 * sgn          # barely a twist -> short tight tail
            _e._burst[0, t] = -0.45             # mass packs INTO the head
    elif stance == 1:                           # Spin: 3 forms (tap 2): vortex -> sawblade -> galaxy
        sm = ctrl["spin_mode"]
        sgn = spin_sign if spin_sign != 0 else 1
        if sm == 0:                             # vortex: the classic compact fast spin
            _e._spin[0, t] = 1.7 * spin_sign
            _e._burst[0, t] = -0.4
        elif sm == 1:                           # sawblade: dense disc + 8 rotating teeth
            _e._spin[0, t] = 1.6 * sgn
            _e._burst[0, t] = -0.45             # packed disc body
            _e._node_m[0, t] = 8.0              # 8 angular clusters = the teeth
            _e._node_w[0, t] = 0.4 * sgn        # the tooth pattern SWEEPS with the spin
        else:                                   # galaxy: wide slow swirl with winding spiral arms
            _e._spin[0, t] = 1.1 * sgn
            _e._burst[0, t] = 0.35              # spread out -> a broad disc
            _e._node_m[0, t] = 3.0              # 3 arms
            _e._node_k[0, t] = 0.25 * sgn       # arms WIND with radius (logarithmic-spiral look)
            _e._node_w[0, t] = -0.05 * sgn      # slow majestic pattern rotation
    elif stance == 2:                           # Drill: 3 modes (slow/med/fast) — grind vs advance
        m = ctrl["drill_mode"]                  # 0 slow / 1 med / 2 fast (tap 3 to rev)
        DSPIN = (0.3, 0.7, 1.5); DSURGE = (1.0, 2.0, 4.0); DADV = (1.0, 0.62, 0.34)
        sgn = spin_sign if spin_sign != 0 else 1   # the bit always spins
        _e._spin[0, t] = DSPIN[m] * sgn         # faster spin -> harder grind, but looser + slower
        _e._drill[0, t, 0] = float(last_dir[0]) * DADV[m]  # |drill| = advance speed
        _e._drill[0, t, 1] = float(last_dir[1]) * DADV[m]
        if DSURGE[m] > 1.0:                     # the spinning front grinds (chews) harder
            _e._surge[0, t] = DSURGE[m]
    elif stance == 3:                           # Wall: a concentrated bar, horizontal or vertical (tap 4 to flip)
        if ctrl["wall_orient"] == 0:            # horizontal bar = vertical facing
            _e._wall[0, t, 0] = 1.25; _e._wall[0, t, 1] = 0.0
        else:                                   # vertical bar = horizontal facing
            _e._wall[0, t, 0] = 0.0; _e._wall[0, t, 1] = 1.25
        _e._burst[0, t] = -0.9                  # strong inward pull -> a dense solid COLUMN, not a picket line
        _e._armor[0, t] = 0.6                   # a FORMED rampart takes 40% less — holding the line is real defense
    elif stance == 4:                           # Pulse: 3 modes (tap 5 to cycle)
        pm = ctrl["pulse_mode"]
        # Q/E swirl works DURING Pulse (was never set -> the spin
        # keys silently did nothing while pulsing)
        _e._spin[0, t] = 0.9 * spin_sign
        if pm == 0:                             # wave: traveling rings + damage crests (the original)
            ring = math.sin(n * 0.33)
            _e._burst[0, t] = 1.0 if ring > 0 else -0.6
            if ring > 0.5:
                _e._surge[0, t] = 4.0           # surge on each ring's crest
        elif pm == 1:                           # rings: cymatic STANDING rings (Chladni circular mode)
            _e._node_l[0, t] = 12.0             # nodal wavelength — concentric resonance bands (was 14: tighter)
            _e._node_v[0, t] = 0.05             # breathe 2.5x the old fixed 0.02 — visibly pumping
            _e._surge[0, t] = 3.0               # (was 2.0) the resonance hits harder
        elif pm == 2:                           # star: cymatic nodal-diameter mode — a 6-petal figure
            _e._node_l[0, t] = 16.0
            _e._node_m[0, t] = 6.0
            # petal sweep you can actually see (was 0.01 drift);
            # direction follows Q/E like the sawblade's
            _e._node_w[0, t] = 0.05 * (spin_sign if spin_sign != 0 else 1)
            _e._node_v[0, t] = 0.05             # rings pump under the petals
            _e._surge[0, t] = 3.0               # (was 2.0)
        elif pm == 3:                           # lattice: SUPERPOSED Chladni modes — breathing rings x sweeping petals
            sgn = spin_sign if spin_sign != 0 else 1
            _e._node_l[0, t] = 16.0             # radial rings...
            _e._node_m[0, t] = 8.0              # ...crossed with 8 petals
            _e._node_k[0, t] = 0.15 * sgn       # petals wind with radius
            _e._node_w[0, t] = 0.03 * sgn       # whole figure slowly sweeps
            _e._node_v[0, t] = 0.05
            _e._surge[0, t] = 2.5
        elif pm == 4:                           # nova: charge (deep gather) then DETONATE
            ph = n % 144                        # ~2.3s cycle at 60Hz
            if ph < 108:                        # implosion: pack tight, menace builds
                _e._burst[0, t] = -0.9
                _e._spin[0, t] = 0.6 * (spin_sign if spin_sign != 0 else 1)
            else:                               # detonation: shockwave + heavy surge
                _e._burst[0, t] = 1.0
                _e._surge[0, t] = 5.0
        else:                                   # tide: rolling DIRECTIONAL fronts (aimed by last_dir, like Wall)
            _e._tide[0, t, 0] = float(last_dir[0])
            _e._tide[0, t, 1] = float(last_dir[1])
            _e._surge[0, t] = 2.5               # the marching crests hit hard
    elif stance == 5:                           # Doom: violent black-hole implosion
        sgn = spin_sign if spin_sign != 0 else 1
        lvl = ctrl["doom_level"]                # 1x/2x/3x charge (tap 6)
        # The ARMY is the visual cue: it forms a spinning ACCRETION
        # DISK — `_ring` holds an open orbit radius (the black hole's
        # dark centre), the swirl spins it, both growing with charge.
        # The client's lensing/amber-disk shader sits in the hole.
        _e._spin[0, t] = (1.2, 1.8, 2.4)[lvl - 1] * sgn
        # tidal surge kept MODEST (1.5-2x; was 4-6x): gravity is Doom's
        # weapon. Diagnosis behind the cut: conversions clustered at the
        # FRONT, not the horizon — the old surge made the conveyor-fed rim
        # a meat grinder that beat every formation; at ~2x a rebalanced
        # Doom assault lands near parity with a plain attack (measured),
        # so it's a strong commit, not an auto-win.
        _e._surge[0, t] = (1.5, 1.75, 2.0)[lvl - 1]
        _e._doom_pos[0, t] = _e.cursor_pos[0, t].float()         # gravity well at YOUR cursor;
        _mass = float(_e.team_oh[0, t].sum())                     # pull ∝ YOUR mass (real black hole)
        # Disk target radius is MASS-SCALED: fighters pack one-per-cell,
        # so a fixed small radius just saturates back into a solid blob.
        # Solve pi*(r_out^2 - r_in^2) = mass for the band centred on the
        # target with its INNER edge at the rendered horizon (r_in,
        # matching the client's 4.4+2.3*lvl graphic) -> the hole stays
        # open and the whole army becomes the spinning accretion disk.
        _r_in = (6.7, 9.0, 11.3)[lvl - 1]
        _ring_val = 0.5 * (_r_in + (_r_in ** 2 + _mass / 3.14159) ** 0.5)
        _e._ring[0, t] = _ring_val
        _e._ring_ecc[0, t] = 1.0                # full oblate -> the edge-on Gargantua blade
        _frac = _mass / max(1.0, _e.fighters_per_team)
        # ladder (24,40,52), not 24*lvl: the sweep showed flat-falloff 72 held
        # an inescapable grip to ~2R — charge concentrates power NEAR the well
        _e._doom_str[0, t] = (24.0, 40.0, 52.0)[lvl - 1] * _frac ** 1.5
        # FINITE reach (was the full map diagonal, which made Doom
        # inescapable -> an auto-win): ~2.2x the disk radius, so a
        # dispersed or kiting enemy escapes the pull and Doom is a
        # committed finisher, not a vacuum.
        _e._doom_range[0, t] = max(56.0, 2.2 * _ring_val)   # floor 70->56: THE kiting fix (sweep-verified)
        # horizon = the RENDERED hole (fixed per charge level), no longer
        # blob-scaled: tying it to mass made every conversion grow the kill
        # zone — the snowball at the heart of "doom tears through
        # everything". What you see is what devours.
        _e._doom_horizon[0, t] = _r_in * 1.25
        # capture rate scales with YOUR mass (was a flat 0.12): a
        # losing army can no longer hold Doom as an unkillable last
        # stand — the devour dies with the disk, so the bigger blob
        # finally gets to consume it.
        _e._doom_cap[0, t] = 0.09 * _frac ** 0.5    # (0.12 -> 0.09, third pass: the passive drain still out-earned its cost)
    elif stance == 6:                           # Maelstrom: a whirlpool CURRENT (cross-team vorticity)
        sgn = spin_sign if spin_sign != 0 else 1
        mm = ctrl["mael_mode"]                  # 0 undertow / 1 ejecta / 2 shear (tap 7)
        # Your army IS the whirlpool body: the ORIGINAL big loose
        # form — a fast wide orbiting shell, burst pushing the mass
        # out into a broad swirling storm-cloud (no ring, no packed
        # disk; nothing like Doom's silhouette).
        _e._spin[0, t] = 2.0 * sgn
        _e._burst[0, t] = 0.5                   # (0.6->0.5) a touch denser: the rim must GRIND
        _e._surge[0, t] = 1.7                   # the spinning rim finally hits like a weapon —
                                                # sweep-verified: this + str 28 is what flips
                                                # ejecta-vs-Doom2x from losing 83/118 to winning 106/92
        # The CURRENT — Doom's radial devour rotated 90°: enemies
        # near the well are swept TANGENTIALLY off their gradient
        # and entrained into orbit through your storm-cloud; no
        # capture, attrition by ordinary contact combat. Strength
        # scales with YOUR mass — a whittled army stirs a weak eddy.
        _mass = float(_e.team_oh[0, t].sum())
        _frac = _mass / max(1.0, _e.fighters_per_team)
        _e._vortex_pos[0, t] = _e.cursor_pos[0, t].float()
        _e._vortex_sign[0, t] = float(sgn)      # current direction follows Q/E
        # Server-tunable dial (cf. _doom_str). Nerfed from 30/2.0x
        # after play: with the flat falloff it yanked units off the
        # enemy blob from across the arena. Now 22 + 1.5x-blob reach
        # + a SQUARED falloff in the engine = a local current you
        # can see coming and skirt, still lethal to wade through.
        _e._vortex_str[0, t] = 28.0 * _frac ** 0.5
        _e._vortex_range[0, t] = max(60.0, 1.5 * (_mass / 3.14159) ** 0.5)
        # radial component per mode: undertow spirals them inward to
        # the rim, ejecta SHOVES them out (the siege-breaker vs an
        # advancing Doom: -0.45 -> -0.7 so the push actually holds a
        # disk off your shell), shear is pure deflection
        _e._vortex_rad[0, t] = (0.30, -0.7, 0.0)[mm]
    elif stance == 7:                           # Atom: 2 forms (tap 8): orbital -> binary star
        sgn = spin_sign if spin_sign != 0 else 1
        # NOTE: the balance matrix found Atom 0-8 (loses every head-on brawl).
        # Tried surge (backfired — sped the boundary churn the denser foe wins)
        # and density (no help) — both empirically worse/equal. The real cause
        # is mechanical: the fig-8 keeps units orbiting, never consolidating, so
        # any dense mass envelops them. Flagged for a REDESIGN (dev log §29),
        # not a knob — reverted to the original look-defining values.
        if ctrl["atom_mode"] == 0:              # figure-8 electron orbitals
            _e._spin[0, t] = 1.8 * sgn          # orbital speed
            _e._burst[0, t] = 0.4               # a little room so the two lobes form
            _e._fig8[0, t] = 1.0                # flip orbit across the cursor -> figure-8 loops
        else:                                   # binary star: two discs orbiting their barycenter
            _e._spin[0, t] = 1.6 * sgn
            _e._burst[0, t] = 0.45
            _e._fig8[0, t] = 2.0                # engine: rotating-axis, co-rotating lobes
    # stance == 8 (Classic) sets NOTHING: every knob stays at the
    # per-tick zero above, so the army just flows down the gradient
    # and packs around the cursor — the original Liquid War blob.
    # (Human-only: the policy's action space stays at 8 stances so
    # existing checkpoints keep loading.)
    # stance == 8 (Classic) sets NOTHING: every knob stays at the per-tick
    # zero, so the army just flows down the gradient — the original blob.
    # MORPH TRANSIENT (16 ticks): flare OUT for the first half, whip-spin and
    # SNAP IN for the second — the formation change is an event you can see
    # (and the client fires a shockwave + the score answers it).
    if morph > 0:
        ph = morph / 16.0                                   # 1 -> 0 across the switch
        _e._spin[0, t] = float(_e._spin[0, t]) * (1.0 + 1.4 * ph) + 0.6 * ph * (spin_sign or 1)
        flare = 0.9 * ph if ph > 0.5 else -0.8 * (1.0 - ph)
        _e._burst[0, t] = max(-1.0, min(1.2, float(_e._burst[0, t]) + flare))
    # FOLLOW-LUNGE: a moving cursor makes the army CHASE — burst tightens
    # with cursor displacement, so the swarm lunges after your hand instead
    # of drifting behind it (settles the instant you stop).
    _cdy = c0_hist[-1][0] - c0_hist[0][0]
    _cdx = c0_hist[-1][1] - c0_hist[0][1]
    _lunge = min(1.0, (abs(_cdy) + abs(_cdx)) / 12.0)
    if _lunge > 0.05:
        _e._burst[0, t] = max(-1.0, float(_e._burst[0, t]) - 0.35 * _lunge)
    # COHESION: travel hugging the cursor without losing the formation —
    # loose/neutral forms get a small extra inward pull; engineered
    # silhouettes (wall bar, maelstrom shell, atom lobes) keep their shapes
    _b_now = float(_e._burst[0, t])
    if -0.3 < _b_now < 0.3:
        _e._burst[0, t] = _b_now - 0.18
    _base_cs = max(1, round(_e.W / 96))
    # EVERY Doom charge pays a mobility tax now (1x included — it was all
    # upside: full speed + pull + free devour, and the AI camps it). The well
    # is a commitment at any level; outmaneuvering it is the baseline counter.
    if ctrl["stance"] == 5:
        return max(1, round(_base_cs * (0.7, 0.45, 0.3)[ctrl["doom_level"] - 1]))  # steeper 2x/3x tax
    return _base_cs


class Player:
    """One human seat in a room: socket + held-control state + its own send
    queue. LATEST-FRAME-WINS: the room loop never awaits a client send — a
    slow WiFi link just drops stale frames instead of throttling the game for
    everyone (the queue holds at most one pending frame)."""

    def __init__(self, sock: WebSocket, team: int, name: str = "") -> None:
        self.sock = sock
        self.team = team
        self.name = name
        self.color = team % 6               # palette index (0..5) — lobby-pickable
        self.loadout: list[int] = []        # the 3-stance kit, for the lobby card
        self.morph = 0                      # ticks left in the stance-switch transient
        self.prev_key = None                # last (stance, modes) snapshot
        self.ctrl: dict[str, Any] = dict(PLAYER_CTRL)
        self.spin_sign = 1
        self.last_dir = [0, 1]
        self.c0_hist = [[0, 0], [0, 0]]
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        self.dead = False
        self.task: asyncio.Task | None = None
        self._sent_bn = -1                  # last binary frame number sent

    def offer(self, frame) -> None:
        """Queue a frame, evicting a stale unsent one (latest wins)."""
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self.queue.put_nowait(frame)

    async def sender(self) -> None:
        try:
            while True:
                blob, bn, hud = await self.queue.get()
                # a FROZEN engine (countdown, result hold) serves the same
                # cached blob every iteration — sending it 63x/s was a
                # 21 MB/s/client storm of identical bytes. Dedup by frame
                # number; the HUD JSON (countdown digits etc.) still flows.
                # A fresh joiner starts at -1, so they always get the
                # current frame even mid-freeze.
                if bn != self._sent_bn:
                    await self.sock.send_bytes(blob)
                    self._sent_bn = bn
                await self.sock.send_json(hud)
        except Exception:
            self.dead = True


def _rle_grid(flat) -> bytes:
    """Run-length encode an int8 cell grid -> (u8 value, u16 runlen) triples,
    little-endian. Runs >65535 are split. flat is a 1-D numpy int8 array."""
    import numpy as np
    n = flat.size
    if n == 0:
        return b""
    chg = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], chg))
    vals = flat[starts]
    lens = np.diff(np.concatenate((starts, [n])))
    if (lens > 65535).any():                       # split long runs into u16 chunks
        ev, el = [], []
        for v, L in zip(vals.tolist(), lens.tolist()):
            while L > 65535:
                ev.append(v); el.append(65535); L -= 65535
            ev.append(v); el.append(L)
        vals = np.array(ev, dtype=np.int8); lens = np.array(el)
    rle = np.empty((vals.size, 3), dtype=np.uint8)
    rle[:, 0] = vals.view(np.uint8)
    rle[:, 1] = (lens & 0xFF).astype(np.uint8)
    rle[:, 2] = ((lens >> 8) & 0xFF).astype(np.uint8)
    return rle.tobytes()


def _clear_knobs(_e) -> None:
    """Per-tick neutral stance state (humans + AI rewrite what they hold)."""
    _e._surge.fill_(1.0)   # neutral damage mult (in-place: graph input)
    if getattr(_e, "_armor", None) is not None:
        _e._armor.fill_(1.0)
    _e._doom_str.zero_(); _e._doom_horizon.zero_(); _e._doom_cap.zero_(); _e._vortex_str.zero_()
    _e._spin.zero_(); _e._burst.zero_(); _e._drill.zero_(); _e._wall.zero_(); _e._fig8.zero_()
    _e._ring.zero_(); _e._ring_ecc.zero_(); _e._node_l.zero_(); _e._node_m.zero_()
    _e._node_k.zero_(); _e._node_w.zero_(); _e._node_v.zero_(); _e._tide.zero_()


class Room:
    """One live game shared by 1..T human players (the rest are AI seats).

    LAN play: every client opens ``/?room=<name>`` — the first one creates the
    room. NAMED rooms open in a LOBBY: players gather over a live all-AI
    battle (free attract mode), pick a color, set a name and kit, and the
    HOST (first joiner) dials how many AI opponents join before pressing
    START — the engine is rebuilt to exactly humans+AI seats. Solo rooms skip
    the lobby: instant play, like always. A leaver's seat hands back to the
    AI mid-game. The room runs ONE game loop and broadcasts each frame to
    every seat (with per-seat HUD fields layered on)."""

    MAX_SEATS = 6                          # client palette / engine ceiling

    def __init__(self, key: str, mode: str, opponent: str, teams: int,
                 size: str = "full", gamemode: str = "annihilate") -> None:
        self.key = key
        self.mode = mode
        self.opponent = opponent           # kept for the lobby-start rebuild
        self.size = size
        self.gamemode = gamemode           # annihilate | koth (the win condition)
        small = size == "small"            # phone boards: same archetypes, half scale
        # 2026-06-13: DOUBLED the board AREA (384x576 -> 544x816, x2.0 cells;
        # phone 192x288 -> 272x408) while keeping the SAME 8000/2000 fighters
        # — sparser armies, far more room to maneuver and kite. Cursor speed
        # auto-scales with width (round(W/96)) so on-screen feel holds; the
        # well range floors (Doom 56 / Maelstrom 60 cells) are now relatively
        # smaller, which only helps escapes. 2x AREA, not 2x linear (that's 4x
        # cells — units become specks and the cold flood lags); LW_PLAY_* env
        # can push it further.
        self._dims = dict(
            height=272 if small else int(os.environ.get("LW_PLAY_H", "544")),
            width=408 if small else int(os.environ.get("LW_PLAY_W", "816")),
            fighters=2000 if small else int(os.environ.get("LW_PLAY_FIGHTERS", "8000")),
        )
        self.session = GameSession(mode=mode, opponent=opponent, teams=teams, **self._dims)
        self.session.gamemode = gamemode
        if small:
            # small boards run EAGER (they're ~5ms/tick anyway): phone rooms
            # churn constantly (screen sleep / tab switches), and the
            # multi-graph capture/teardown traffic that churn generates is
            # what kept poisoning the CUDA context in async mode. The one
            # long-lived big-board graph has been stable all day.
            self.session.engine._cuda_graph = False
        self.named = not key.startswith("~solo-")
        self.phase = "lobby" if self.named else "play"
        self.host: int | None = None       # seat of the first joiner — starts the match
        self.ai_count = 1                  # host dial: AI seats folded in at START
        self.start_flag = False
        self.players: dict[int, Player] = {}
        self.recent: dict[str, tuple[int, int, float]] = {}  # name -> (seat, color, left_at)
        self.task: asyncio.Task | None = None
        self.closed = False
        self.map_choice: int | None = None
        self.reset_flag = False
        self.notes: list[list] = []         # queued join/leave toasts [payload, ttl]
        self.wins: dict[int, int] = {}      # session scoreboard, by seat
        # solo round 1 gets its 3-2-1 too (it used to start mid-stride)
        self.freeze = 3 * int(TICK_HZ) if self.phase == "play" else 0
        self.overtime = False

    def note_push(self, payload: dict) -> None:
        """Queue a toast — back-to-back joins used to overwrite each other."""
        self.notes.append([payload, 90])
        del self.notes[:-4]

    def _palette(self) -> list[int]:
        """Per-TEAM palette indices (0..5) for the client: humans wear their
        picked color, AI seats dress from the leftovers — color is COSMETIC
        and travels with the player, decoupled from seat index."""
        T = self.session.engine.T
        cols = [-1] * T
        taken = set()
        for t, p in self.players.items():
            taken.add(p.color)
            if t < T:
                cols[t] = p.color
        pool = [c for c in range(6) if c not in taken]
        for t in range(T):
            if cols[t] < 0:
                cols[t] = pool.pop(0) if pool else t % 6
        return cols

    def join(self, sock: WebSocket, name: str = "",
             color: int | None = None) -> Player | None:
        # the lobby seats up to MAX_SEATS — START compacts everyone onto the
        # engine; mid-match joins are capped by the live engine's seat count
        cap = self.MAX_SEATS if self.phase == "lobby" else self.session.engine.T
        free = [t for t in range(cap) if t not in self.players]
        if not free:
            return None
        # REJOIN GRACE: a name that left <2 min ago gets its seat + color back
        # without the fair-join restart — wifi blips, phone sleep and fat-
        # fingered tab closes shouldn't cost the table the match.
        back = self.recent.pop(name.lower(), None) if name else None
        rejoin = back is not None and time.monotonic() - back[2] < 120
        seat = back[0] if rejoin and back[0] in free else free[0]
        p = Player(sock, seat, name)
        taken = {pl.color for pl in self.players.values()}
        pool = [c for c in range(6) if c not in taken]
        want = back[1] if rejoin else color
        p.color = want if want in pool else (pool[0] if pool else seat % 6)
        p.task = asyncio.create_task(p.sender())
        had_humans = bool(self.players)
        self.players[seat] = p
        if self.host is None or self.host not in self.players:
            self.host = seat
        # FAIR JOIN: landing mid-match means inheriting whatever shape the AI
        # left that army in. Past a short grace window, a joiner triggers a
        # fresh round (maps are point-symmetric — restarts are fair). Lobby
        # joins and rejoins just sit down.
        if (self.phase == "play" and not rejoin and had_humans
                and (self.session.engine.tick > 8 * TICK_HZ or self.session.done)):
            self.reset_flag = True
            self.note_push({"ev": "join_restart", "team": seat})
        else:
            self.note_push({"ev": "join", "team": seat})
        return p

    def leave(self, team: int) -> None:
        p = self.players.pop(team, None)    # the seat reverts to AI control
        if p is not None and p.task is not None:
            p.task.cancel()
        if p is not None and p.name:
            self.recent[p.name.lower()] = (team, p.color, time.monotonic())
        if self.players:
            self.note_push({"ev": "leave", "team": team})
            if self.host == team:           # the crown passes to the next seat
                self.host = min(self.players)
        if not self.players:
            self.closed = True

    def _start_match(self) -> None:
        """The host pressed START: size the engine to the lobby — humans get
        seats 0..n-1 (colors travel with them) plus the host's chosen AI
        seats — then open with the 3-2-1 countdown."""
        humans = sorted(self.players)
        want = max(2, min(self.MAX_SEATS, len(humans) + self.ai_count))
        e = self.session.engine
        if want != e.T:
            if getattr(e, "_graph", None) is not None:  # same teardown hazard as Room.run
                torch.cuda.synchronize()
                e._graph = None
            self.session = GameSession(mode=self.mode, opponent=self.opponent,
                                       teams=want, **self._dims)
            self.session.gamemode = self.gamemode
            if self.size == "small":
                self.session.engine._cuda_graph = False
        old = dict(self.players)
        self.players.clear()
        new_wins: dict[int, int] = {}
        for new_t, old_t in enumerate(sorted(old)):
            pl = old[old_t]
            pl.team = new_t
            self.players[new_t] = pl
            if self.host == old_t:
                self.host = new_t
            if old_t in self.wins:          # humans keep their tallies; AI seats
                new_wins[new_t] = self.wins[old_t]   # are new identities anyway
        self.wins = new_wins
        self.session.engine._map_choice = self.map_choice
        self.session.reset()
        _clear_knobs(self.session.engine)
        self.reset_flag = False
        self.freeze = 3 * int(TICK_HZ)
        self.overtime = False
        self.session._overtime = 1.0
        self.phase = "play"
        self.note_push({"ev": "start"})

    async def run(self) -> None:
        try:
            await self._run()
        except Exception:
            import traceback
            print(f"[room {self.key}] game loop crashed:\n{traceback.format_exc()}", flush=True)
        finally:
            self.closed = True
            # NEVER let a captured CUDA graph be GC'd with replays in flight:
            # room teardown (phone sleeps, tab closes) lands ~16ms after the
            # last replay was enqueued, and freeing the graph's pool under
            # running kernels poisons the whole CUDA context (every later
            # session then dies at construction). Same race as mid-game
            # reset — this is the other door.
            if getattr(self.session.engine, "_graph", None) is not None:
                torch.cuda.synchronize()
                self.session.engine._graph = None

    async def _run(self) -> None:
        dt = 1.0 / TICK_HZ
        hold = 0                                       # ticks to linger on a finished game
        fps = TICK_HZ                                  # achieved frame rate (EMA)
        prev = None
        logged = False                                 # one telemetry line per finished game
        loop = asyncio.get_event_loop()
        n = 0
        next_dl = loop.time()                          # absolute frame deadline (drift-corrected)

        while not self.closed:
            t0 = loop.time()                                  # frame start, for steady pacing
            session = self.session                            # _start_match may rebuild it
            players = list(self.players.values())
            for p in players:
                if p.ctrl["spin"] is not None:                # Q/E -> orbit direction
                    p.spin_sign = p.ctrl["spin"]; p.ctrl["spin"] = None
                if p.ctrl["map"] is not None:
                    self.map_choice = p.ctrl["map"] if p.ctrl["map"] >= 0 else None
                    p.ctrl["map"] = None
                if p.ctrl["reset"]:
                    self.reset_flag = True; p.ctrl["reset"] = False
            if self.phase == "lobby":
                # GATHER: the lobby card floats over a live all-AI battle.
                # Player input waits; the host's START flips to play.
                if self.start_flag:
                    self.start_flag = False
                    self._start_match()
                    session = self.session
                    hold = 0; logged = False
                elif session.done or self.reset_flag:
                    session.engine._map_choice = self.map_choice
                    session.reset(); self.reset_flag = False
                    _clear_knobs(session.engine)
                else:
                    _clear_knobs(session.engine)
                    session._overtime = 1.0
                    session.step({})                   # every seat AI-driven
                    _ea = session.engine               # attract pays the same sweep cap
                    if (_ea.tick >= 80 and getattr(_ea, "_fixed_sweeps", None) is None
                            and not getattr(_ea, "_cuda_graph", False)):
                        _ea._fixed_sweeps = int(os.environ.get("LW_GRAD_SWEEPS", str(_ea.cursor_speed + 4)))
            elif self.reset_flag:
                session.engine._map_choice = self.map_choice   # picked map (None=random)
                session.reset(); self.reset_flag = False; hold = 0; logged = False
                _clear_knobs(session.engine)
                self.freeze = 3 * int(TICK_HZ)         # 3-2-1-GO: everyone starts on GO
                self.overtime = False                  # last round's surge isn't this round's
                session._overtime = 1.0
            elif session.match_over:                   # annihilation OR the hill was won
                hold += 1
                # POST-MATCH MOMENT: hold the result for ~20s — long enough to
                # read the card and hit Rematch (any {reset} ends it early).
                # PRACTICE: no result card, just a short beat then the NEXT
                # lesson (the dummy's stance rotates) — endless low-stakes reps.
                if hold > TICK_HZ * (4 if session.practice else 20):
                    if session.practice:
                        session._lesson += 1
                    session.engine._map_choice = self.map_choice
                    session.reset(); hold = 0; logged = False
                    _clear_knobs(session.engine)
                    self.freeze = 3 * int(TICK_HZ)
                    self.overtime = False
                    session._overtime = 1.0
            elif self.freeze > 0:
                self.freeze -= 1                       # armies hold their breath
            else:
                _e = session.engine
                _clear_knobs(_e)
                base_cs = max(1, round(_e.W / 96))
                speeds = [base_cs] * _e.T
                humans = {}
                for p in players:
                    if p.ctrl["dir"] and (p.ctrl["dir"][0] or p.ctrl["dir"][1]):
                        p.last_dir = p.ctrl["dir"]            # heading the Drill/Wall point at
                    # stance OR mode change kicks the 16-tick morph transient
                    key = (p.ctrl["stance"], p.ctrl["drill_mode"], p.ctrl["doom_level"],
                           p.ctrl["wall_orient"], p.ctrl["pulse_mode"], p.ctrl["spin_mode"],
                           p.ctrl["mael_mode"], p.ctrl["swarm_mode"], p.ctrl["atom_mode"])
                    if key != p.prev_key:
                        if p.prev_key is not None:
                            p.morph = 16
                        p.prev_key = key
                    speeds[p.team] = _apply_player_stance(
                        _e, p.team, p.ctrl, p.spin_sign, p.last_dir, p.c0_hist, n,
                        morph=p.morph)
                    if p.morph > 0:
                        p.morph -= 1
                    humans[p.team] = (p.ctrl["target"], p.ctrl["dir"])
                _e._cursor_speed_t = speeds                   # per-seat (Doom slows ITS holder only)
                # OVERTIME SURGE: equal-mass conversion combat can trench-war
                # forever. Past 3:00, damage escalates (compounding per minute,
                # capped 2.5x) so games CONCLUDE through normal play.
                ot_min = (session.engine.tick / TICK_HZ - 180.0) / 60.0
                self.overtime = ot_min > 0
                session._overtime = min(2.5, 1.0 + 0.15 * ot_min) if self.overtime else 1.0
                session.step(humans)
                # past the cold flood, pin the eager gradient to the same fixed
                # sweep count the captured graph used (cursor_speed+4): the
                # convergence early-out never fires with a moving cursor anyway,
                # and dropping it removes ~516 launches + 12 sync stalls/tick
                if (_e.tick >= 80 and getattr(_e, "_fixed_sweeps", None) is None
                        and not getattr(_e, "_cuda_graph", False)):
                    _e._fixed_sweeps = int(os.environ.get("LW_GRAD_SWEEPS", str(_e.cursor_speed + 4)))
            blob = session.frame_blob()
            if blob is not getattr(self, "_last_sent_blob", None):
                self._last_sent_blob = blob
                self._blob_n = getattr(self, "_blob_n", 0) + 1
            st = session.state(); st["fps"] = round(fps, 1)
            st["players"] = len(players)
            st["seats"] = sorted(self.players)            # which teams are humans (lobby chips)
            st["names"] = {str(t): pl.name for t, pl in self.players.items() if pl.name}
            st["phase"] = self.phase
            st["ai_n"] = self.ai_count
            st["colors"] = self._palette()                # cosmetic palette map, lobby-picked
            if session.practice:                          # forgiving sandbox: coach line, no card
                st["practice"] = True
                st["lesson"] = LESSONS[session._lesson % len(LESSONS)][1]
            if self.host is not None:
                st["host"] = self.host
            if self.phase == "lobby":
                st["lobby"] = [{"team": t, "name": pl.name, "color": pl.color,
                                "loadout": pl.loadout[:3]}
                               for t, pl in sorted(self.players.items())]
            if self.wins:
                st["wins"] = {str(t): n for t, n in self.wins.items()}
            if self.freeze > 0:
                st["countdown"] = (self.freeze + int(TICK_HZ) - 1) // int(TICK_HZ)
            if getattr(self, "overtime", False) and not session.match_over:
                st["overtime"] = True
            if session.gamemode == "koth":             # objective HUD: hill + score bars
                st["gamemode"] = "koth"
                st["hill"] = list(session.hill)        # (cy, cx, r)
                st["koth"] = session.koth_score
                st["koth_target"] = session.koth_target
            if session.match_over:
                st["done"] = True                      # koth win sets done even with both alive
                st["winner"] = session.winner()
            else:
                # ROUT BEACON: a team under 4% of starting mass broadcasts its
                # remnant centroid — endgame drag was never killing the last
                # drops, it was FINDING them on a 384x576 board. (The loop var
                # must NOT be ``n`` — it shadowed the room's pulse clock and
                # froze human Pulse wave/nova timing on a fighter count.)
                e2 = session.engine
                routs = []
                for t, cnt in enumerate(st["fighters"]):
                    if 0 < cnt < 0.04 * e2.fighters_per_team:
                        m = (e2.fteam[0] == t)
                        if m.any():        # HUD counts are ~6 ticks stale: no NaN centroids
                            routs.append([t, float(e2.fy[0, m].float().mean()),
                                          float(e2.fx[0, m].float().mean())])
                if routs:
                    st["rout"] = routs
            if self.notes:
                st["note"] = self.notes[0][0]
                self.notes[0][1] -= 1
                if self.notes[0][1] <= 0:
                    self.notes.pop(0)
            for p in players:
                if p.team < len(st["cursors"]):    # lobby seats can outnumber engine seats
                    p.c0_hist.append(list(st["cursors"][p.team])); del p.c0_hist[:-7]   # comet aim window
            if self.phase == "play" and session.match_over and not logged and not session.practice:
                w = session.winner()
                self.wins[w] = self.wins.get(w, 0) + 1
                print(f"[telemetry] GAME END room={self.key} mode={session.gamemode} map={st['map']} tick={st['tick']} "
                      f"winner=team{w} counts={st['fighters']} koth={session.koth_score} fps={fps:.1f}", flush=True)
                logged = True
            elif not session.match_over and st["tick"] and st["tick"] % 150 == 0:
                print(f"[telemetry] tick={st['tick']} fps={fps:.1f} flood={st['flood_pct']}% "
                      f"spread={st['spread']} counts={st['fighters']}", flush=True)
            # one shared binary frame + per-seat HUD JSON, via each player's
            # latest-wins queue — the room loop NEVER blocks on a slow client
            for p in players:
                if p.dead:
                    self.leave(p.team)
                    continue
                msg = dict(st)
                msg["you"] = p.team
                msg["stance"] = STANCES[p.ctrl["stance"]]
                msg["mode"] = _mode_name(p.ctrl)
                msg["spin_dir"] = p.spin_sign
                p.offer((blob, self._blob_n, msg))
            t_work = loop.time()
            next_dl += dt                                # advance the absolute deadline
            sleep = next_dl - loop.time()
            if sleep < -dt:                              # fell far behind -> resync (don't spiral)
                next_dl = loop.time(); sleep = 0.0
            await asyncio.sleep(max(0.0, sleep))
            now = loop.time()
            if prev is not None and now > prev:        # measure actual loop period
                fps = 0.9 * fps + 0.1 / (now - prev)
            if n % 120 == 0:
                print(f"[perf] room={self.key} players={len(players)} "
                      f"work={(t_work - t0) * 1000:.1f}ms period={(now - t0) * 1000:.1f}ms fps={fps:.0f}", flush=True)
            prev = now
            n += 1                                     # advance the pulse clock


ROOMS: dict[str, Room] = {}


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    """Room-based play: ``room=<name>`` shares one game between players (LAN
    multiplayer — open the same room name on another machine to join as the
    next team); no room param = a private solo room. One server-driven game
    loop per room; per-socket receivers feed each seat's held controls."""
    await sock.accept()
    q = sock.query_params
    mode = q.get("mode", "play")
    key = q.get("room") or f"~solo-{id(sock)}"
    if mode == "spectate":
        # spectators never occupy a seat in a shared room (they'd freeze an
        # army and trigger fair-join restarts) — spectate is always solo
        key = f"~solo-{id(sock)}"
    room = ROOMS.get(key)
    if room is None or room.closed:
        size = q.get("size", "full")
        teams = int(q.get("teams", "2"))
        if not key.startswith("~solo-"):
            # NAMED rooms default to family size — empty seats are AI until a
            # human claims one, so the bigger board costs nothing socially and
            # the third player stops bouncing off a silently-full 1v1.
            floor = int(os.environ.get("LW_ROOM_SEATS", "4" if size == "small" else "3"))
            teams = max(teams, floor)
        gm = q.get("gm", "annihilate")
        room = Room(key, mode=mode,
                    opponent=q.get("opponent", "latest"),
                    teams=teams, size=size,
                    gamemode=gm if gm in ("annihilate", "koth") else "annihilate")
        ROOMS[key] = room
    colorq = q.get("color")
    player = room.join(sock, (q.get("name") or "").strip()[:12],
                       int(colorq) if colorq and colorq.isdigit() else None)
    if player is None:                                  # room full: say so, don't ghost
        await sock.send_json({"error": "room_full", "room": key,
                              "seats": room.session.engine.T,
                              "players": len(room.players)})
        await sock.close(code=4001)
        return
    if room.task is None:
        room.task = asyncio.create_task(room.run())
    ctrl = player.ctrl
    try:
        while True:
            msg = await sock.receive_json()
            if msg.get("reset"):
                ctrl["reset"] = True
            elif "map" in msg:                 # map picker -> force archetype + new game
                m = msg["map"]
                ctrl["map"] = -1 if m is None else int(m)
                ctrl["reset"] = True
            elif "spin" in msg:                # Q/E -> orbit direction (Spin/Swarm stances)
                ctrl["spin"] = float(msg["spin"])
            elif "stance" in msg:              # number row -> held stance; re-tap cycles its mode
                s = max(0, min(8, int(msg["stance"])))
                if s == 2 and ctrl["stance"] == 2:
                    ctrl["drill_mode"] = (ctrl["drill_mode"] + 1) % 3
                elif s == 2:
                    ctrl["drill_mode"] = 0
                if s == 5 and ctrl["stance"] == 5:   # re-tap Doom charges the gravity 1x -> 2x -> 3x
                    ctrl["doom_level"] = ctrl["doom_level"] % 3 + 1
                elif s == 5:
                    ctrl["doom_level"] = 1
                if s == 3 and ctrl["stance"] == 3:    # re-tap Wall flips the bar horizontal <-> vertical
                    ctrl["wall_orient"] ^= 1
                if s == 4 and ctrl["stance"] == 4:    # re-tap Pulse cycles its 6 modes
                    ctrl["pulse_mode"] = (ctrl["pulse_mode"] + 1) % 6
                elif s == 4:
                    ctrl["pulse_mode"] = 0
                if s == 1 and ctrl["stance"] == 1:    # re-tap Spin cycles vortex -> sawblade -> galaxy
                    ctrl["spin_mode"] = (ctrl["spin_mode"] + 1) % 3
                elif s == 1:
                    ctrl["spin_mode"] = 0
                if s == 6 and ctrl["stance"] == 6:    # re-tap Maelstrom cycles undertow -> ejecta -> shear
                    ctrl["mael_mode"] = (ctrl["mael_mode"] + 1) % 3
                elif s == 6:
                    ctrl["mael_mode"] = 0
                if s == 0 and ctrl["stance"] == 0:    # re-tap Swarm cycles web -> comet
                    ctrl["swarm_mode"] ^= 1
                if s == 7 and ctrl["stance"] == 7:    # re-tap Atom cycles orbital -> binary star
                    ctrl["atom_mode"] ^= 1
                ctrl["stance"] = s
            elif "name" in msg:                # lobby: live rename
                player.name = str(msg["name"]).strip()[:12]
            elif "color" in msg:               # lobby: pick a color (if free)
                try:
                    c = int(msg["color"])
                except (TypeError, ValueError):
                    continue
                if 0 <= c < 6 and all(pl.color != c for t2, pl in room.players.items()
                                      if t2 != player.team):
                    player.color = c
            elif "loadout" in msg:             # the 3-stance kit, for the lobby card
                try:
                    player.loadout = [int(x) for x in list(msg["loadout"])[:3]
                                      if 0 <= int(x) <= 8]
                except (TypeError, ValueError):
                    pass
            elif "ai" in msg:                  # host dial: AI opponents at START
                try:
                    if player.team == room.host:
                        room.ai_count = max(0, min(5, int(msg["ai"])))
                except (TypeError, ValueError):
                    pass
            elif msg.get("start"):             # host opens the match
                if player.team == room.host and room.phase == "lobby":
                    room.start_flag = True
            elif msg.get("lobby"):             # back to the gather screen
                if room.named:
                    room.phase = "lobby"
            elif "dir" in msg:                 # keyboard (arrows/WASD)
                ctrl["dir"] = msg["dir"]
            elif "target" in msg:              # mouse
                ctrl["target"] = msg["target"]
    except WebSocketDisconnect:
        pass
    finally:
        room.leave(player.team)
        # pop only OUR room: a successor room created under the same key
        # while this socket lingered must not be evicted from the index
        if room.closed and ROOMS.get(key) is room:
            ROOMS.pop(key, None)


@app.get("/rooms")
async def rooms() -> list[dict[str, Any]]:
    """Open named rooms, for the zero-typing 'Join a game' list. ``async`` so
    it runs ON the event loop — the threadpool version raced live joins."""
    out = []
    for key, r in list(ROOMS.items()):
        if key.startswith("~solo-") or r.closed or not r.players:
            continue
        e = r.session.engine
        plist = list(r.players.values())
        out.append({"room": key, "players": len(plist),
                    "seats": Room.MAX_SEATS if r.phase == "lobby" else e.T,
                    "phase": r.phase,
                    "names": [p.name or f"P{p.team + 1}" for p in plist],
                    "map": MAP_NAMES[e._last_arch] if 0 <= getattr(e, "_last_arch", -1) < len(MAP_NAMES) else "?",
                    "tick": int(e.tick)})
    return out


@app.get("/checkpoints")
def checkpoints() -> list[dict[str, Any]]:
    """The opponent roster: every checkpoint under CKPT_DIR (the live
    ``rl/best/policy.pt`` plus the promotion job's datestamped archive),
    newest first. The client populates the opponent dropdown from this."""
    root = Path(CKPT_DIR)
    seen = {}
    for pat in ("rl/*/*.pt", "*/*.pt"):
        for f in glob.glob(str(root / pat)):
            seen[f] = os.path.getmtime(f)
    return [{"id": str(Path(f).relative_to(root)),
             "name": Path(f).stem,
             "mtime": int(m)}
            for f, m in sorted(seen.items(), key=lambda kv: -kv[1])]


@app.get("/liquidwar.apk")
def apk() -> FileResponse:
    """The Android app, served as an explicit DOWNLOAD (Content-Disposition).
    Chrome on Android refuses 'dangerous' file types over plain-HTTP unless
    the response is unambiguous about being an attachment — and even then it
    warns; the sheet links a .zip fallback that downloads without the fuss."""
    f = _STATIC / "liquidwar.apk"
    if not f.is_file():
        raise HTTPException(404, "APK not built yet")
    return FileResponse(f, media_type="application/vnd.android.package-archive",
                        filename="liquidwar.apk",
                        headers={"Content-Disposition": 'attachment; filename="liquidwar.apk"'})


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    ck = _latest_checkpoint()
    return {"ok": True, "device": DEVICE, "tick_hz": TICK_HZ,
            "latest_ckpt": ck and Path(ck).name}


if _STATIC.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC), html=True), name="static")
