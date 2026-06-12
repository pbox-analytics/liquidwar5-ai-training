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
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from rl.eval import _heuristic_dydx
from rl.policy import LEGACY_ACTION, NUM_STANCES, CursorPolicy, act, apply_stances
from simulator.engine import LiquidWarEngine, GRADIENT_INF, MAP_NAMES

CKPT_DIR = os.environ.get("LW_CKPT_DIR", "/opt/training/results")  # NFS mount
DEVICE = os.environ.get("LW_PLAY_DEVICE", "cpu")
TICK_HZ = float(os.environ.get("LW_TICK_HZ", "60"))               # game + frame rate (60fps;
#   the deploy over-targets 63 so asyncio's ~1ms sleep granularity lands on a true 60)
_STATIC = Path(__file__).parent / "static"


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
        self.ckpt_name = opponent
        if opponent not in ("heuristic", "random"):
            if opponent == "latest":
                path = _latest_checkpoint()
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

    @torch.no_grad()
    def _ai_dydx(self):
        """``(dydx, ai_stance)`` for ALL teams from the configured AI. ``ai_stance``
        is the policy's chosen stance per team (``None`` for the heuristic, which
        has no stances)."""
        if self.policy is None:
            return _heuristic_dydx(self.engine), None
        obs = self.engine.get_observation()
        dydx, stance, _, _, _ = act(self.policy, obs, self.engine.T,
                                    self.engine.team_alive, deterministic=True)
        if getattr(self, "_legacy", None) is not None:    # old 8-stance policy.pt
            stance = self._legacy[stance]
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
            apply_stances(self.engine, ai_stance, dydx, human_teams=human_set)
            self._ai_fx_markers(ai_stance, human_set)
            # AI Doom holders glide slow too (parity with the human dial)
            spd = getattr(self.engine, "_cursor_speed_t", None)
            if spd is not None:
                base = max(1, round(self.engine.W / 96))
                acts = ai_stance[0].tolist()
                for t in range(self.engine.T):
                    if t not in human_set:
                        a = acts[t]
                        spd[t] = (max(1, round(base * (0.7, 0.5, 0.35)[a - 16]))
                                  if 16 <= a <= 18 else base)
        self.engine.step(dydx)

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
                self._ai_mael = [t, *e.cursor_pos[0, t].tolist()]

    def reset(self) -> None:
        self.engine.reset()
        self._prev_pos = None                  # force a keyframe after every reset

    def frame_blob(self) -> bytes:
        """The per-tick BINARY channel: mote positions + teams (+ the cell
        grid at ~5 Hz). Fighters move <= unit_speed (6) cells/tick, so between
        periodic int16 KEYFRAMES every position update is an int8 DELTA —
        about half the bytes of the old base64-in-JSON stream, with zero
        client-side atob/JSON cost. Layout (little-endian):
          u8 type (1=keyframe int16 abs, 2=delta int8) | u8 hasGrid |
          u16 pn | u32 tick | pos[2*pn] | pteam u8[pn] | grid i8[H*W]?
        """
        e = self.engine
        step = max(1, e.N // 9000)
        pidx = torch.arange(0, e.N, step, device=e.device)
        pos = torch.stack((e.fy[0, pidx], e.fx[0, pidx]), dim=1).reshape(-1).to(torch.int16).cpu()
        prev = getattr(self, "_prev_pos", None)
        key = prev is None or prev.numel() != pos.numel() or int(e.tick) % 30 == 0
        self._prev_pos = pos
        body = (pos if key else (pos - prev).to(torch.int8)).numpy().tobytes()
        pteam = e.fteam[0, pidx].to(torch.uint8).cpu().numpy().tobytes()
        send_grid = (int(e.tick) % 12 == 0) or int(e.tick) < 8 or self.done
        grid = b""
        if send_grid:
            oh = e.team_oh[0]
            present = oh.sum(0) > 0
            cell = oh.argmax(0).to(torch.int8)
            cell = torch.where(present, cell, torch.full_like(cell, -1))
            cell = torch.where(e.walls[0], torch.full_like(cell, -2), cell)  # -2 = wall
            grid = cell.cpu().numpy().tobytes()
        import struct
        head = struct.pack("<BBHI", 1 if key else 2, 1 if send_grid else 0,
                           pidx.numel(), int(e.tick))
        return head + body + pteam + grid

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
        if not hasattr(self, "_hud") or int(e.tick) % 6 == 0:
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
            else ("cloud", "comet")[ctrl["swarm_mode"]] if ctrl["stance"] == 0
            else ("orbital", "binary")[ctrl["atom_mode"]] if ctrl["stance"] == 7 else "")


def _apply_player_stance(_e, t, ctrl, spin_sign, last_dir, c0_hist, n) -> int:
    """Apply ONE human player's held stance to team ``t``'s knobs — the exact
    block that used to be inlined for team 0, parametrized by seat so any
    number of humans can hold stances in the same game. Returns the player's
    cursor speed (Doom charge trades speed for pull)."""
    stance = ctrl["stance"]                     # 0 Swarm 1 Spin 2 Drill 3 Wall 4 Pulse
    if stance == 0:                             # Swarm: 2 forms (tap 1): cloud -> comet
        if ctrl["swarm_mode"] == 0:             # cloud: loose, varied-radius orbits
            _e._spin[0, t] = 0.5 * spin_sign
            _e._burst[0, t] = 0.15              # slight loosen -> a diffuse orbiting cloud
        else:                                   # comet: a teardrop along your MOTION
            sgn = spin_sign if spin_sign != 0 else 1
            # aim = recent cursor displacement (works for mouse AND
            # keys, unlike last_dir); standing still relaxes the
            # comet back into a blob
            cdy = (c0_hist[-1][0] > c0_hist[0][0]) - (c0_hist[-1][0] < c0_hist[0][0])
            cdx = (c0_hist[-1][1] > c0_hist[0][1]) - (c0_hist[-1][1] < c0_hist[0][1])
            _e._drill[0, 0, 0] = 0.85 * cdy     # dense head pierces along the motion
            _e._drill[0, 0, 1] = 0.85 * cdx     #   (the drill machinery, velocity-aimed)
            _e._spin[0, t] = 0.35 * sgn         # slight twist gives the tail life
            _e._burst[0, t] = -0.25             # packed head, trailing wake
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
        _e._drill[0, 0, 0] = float(last_dir[0]) * DADV[m]  # |drill| = advance speed
        _e._drill[0, 0, 1] = float(last_dir[1]) * DADV[m]
        if DSURGE[m] > 1.0:                     # the spinning front grinds (chews) harder
            _e._surge[0, t] = DSURGE[m]
    elif stance == 3:                           # Wall: a concentrated bar, horizontal or vertical (tap 4 to flip)
        if ctrl["wall_orient"] == 0:            # horizontal bar = vertical facing
            _e._wall[0, 0, 0] = 1.0; _e._wall[0, 0, 1] = 0.0
        else:                                   # vertical bar = horizontal facing
            _e._wall[0, 0, 0] = 0.0; _e._wall[0, 0, 1] = 1.0
        _e._burst[0, t] = -0.9                  # strong inward pull -> a dense solid COLUMN, not a picket line
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
            _e._tide[0, 0, 0] = float(last_dir[0])
            _e._tide[0, 0, 1] = float(last_dir[1])
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
        _e._doom_str[0, t] = lvl * 24.0 * _frac ** 1.5  # (32->24, parity rebalance) super-linear in mass: peels the loosely-bound periphery, not the whole army (x1/x2/x3 charge, tap 6)
        # FINITE reach (was the full map diagonal, which made Doom
        # inescapable -> an auto-win): ~2.2x the disk radius, so a
        # dispersed or kiting enemy escapes the pull and Doom is a
        # committed finisher, not a vacuum.
        _e._doom_range[0, t] = max(70.0, 2.2 * _ring_val)
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
        _e._burst[0, t] = 0.6
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
        _e._vortex_str[0, t] = 22.0 * _frac ** 0.5
        _e._vortex_range[0, t] = max(60.0, 1.5 * (_mass / 3.14159) ** 0.5)
        # radial component per mode: undertow spirals them inward to
        # the rim, ejecta SHOVES them out (the siege-breaker vs an
        # advancing Doom: -0.45 -> -0.7 so the push actually holds a
        # disk off your shell), shear is pure deflection
        _e._vortex_rad[0, t] = (0.30, -0.7, 0.0)[mm]
    elif stance == 7:                           # Atom: 2 forms (tap 8): orbital -> binary star
        sgn = spin_sign if spin_sign != 0 else 1
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
    _base_cs = max(1, round(_e.W / 96))
    # EVERY Doom charge pays a mobility tax now (1x included — it was all
    # upside: full speed + pull + free devour, and the AI camps it). The well
    # is a commitment at any level; outmaneuvering it is the baseline counter.
    if ctrl["stance"] == 5:
        return max(1, round(_base_cs * (0.7, 0.5, 0.35)[ctrl["doom_level"] - 1]))
    return _base_cs


class Player:
    """One human seat in a room: socket + held-control state + its own send
    queue. LATEST-FRAME-WINS: the room loop never awaits a client send — a
    slow WiFi link just drops stale frames instead of throttling the game for
    everyone (the queue holds at most one pending frame)."""

    def __init__(self, sock: WebSocket, team: int) -> None:
        self.sock = sock
        self.team = team
        self.ctrl: dict[str, Any] = dict(PLAYER_CTRL)
        self.spin_sign = 1
        self.last_dir = [0, 1]
        self.c0_hist = [[0, 0], [0, 0]]
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        self.dead = False
        self.task: asyncio.Task | None = None

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
                blob, hud = await self.queue.get()
                await self.sock.send_bytes(blob)
                await self.sock.send_json(hud)
        except Exception:
            self.dead = True


class Room:
    """One live game shared by 1..T human players (the rest are AI seats).

    LAN play: every client opens ``/?room=<name>`` — the first one creates the
    room (their mode/opponent/teams params win), later ones take the next free
    seat. A leaver's seat hands back to the AI mid-game. The room runs ONE
    game loop and broadcasts each frame to every seat (with per-seat HUD
    fields layered on)."""

    def __init__(self, key: str, mode: str, opponent: str, teams: int,
                 size: str = "full") -> None:
        self.key = key
        small = size == "small"            # phone boards: same archetypes, half scale
        self.session = GameSession(
            mode=mode, opponent=opponent, teams=teams,
            height=192 if small else int(os.environ.get("LW_PLAY_H", "384")),
            width=288 if small else int(os.environ.get("LW_PLAY_W", "576")),
            fighters=2000 if small else int(os.environ.get("LW_PLAY_FIGHTERS", "8000")),
        )
        if small:
            # small boards run EAGER (they're ~5ms/tick anyway): phone rooms
            # churn constantly (screen sleep / tab switches), and the
            # multi-graph capture/teardown traffic that churn generates is
            # what kept poisoning the CUDA context in async mode. The one
            # long-lived big-board graph has been stable all day.
            self.session.engine._cuda_graph = False
        self.players: dict[int, Player] = {}
        self.task: asyncio.Task | None = None
        self.closed = False
        self.map_choice: int | None = None
        self.reset_flag = False

    def join(self, sock: WebSocket) -> Player | None:
        free = [t for t in range(self.session.engine.T) if t not in self.players]
        if not free:
            return None
        p = Player(sock, free[0])
        p.task = asyncio.create_task(p.sender())
        self.players[p.team] = p
        return p

    def leave(self, team: int) -> None:
        p = self.players.pop(team, None)    # the seat reverts to AI control
        if p is not None and p.task is not None:
            p.task.cancel()
        if not self.players:
            self.closed = True

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
        session = self.session
        dt = 1.0 / TICK_HZ
        hold = 0                                       # ticks to linger on a finished game
        fps = TICK_HZ                                  # achieved frame rate (EMA)
        prev = None
        logged = False                                 # one telemetry line per finished game
        loop = asyncio.get_event_loop()
        n = 0
        next_dl = loop.time()                          # absolute frame deadline (drift-corrected)

        def clear_knobs(_e) -> None:
            _e._surge.fill_(1.0)   # neutral damage mult (in-place: graph input)
            _e._doom_str.zero_(); _e._doom_horizon.zero_(); _e._doom_cap.zero_(); _e._vortex_str.zero_()
            _e._spin.zero_(); _e._burst.zero_(); _e._drill.zero_(); _e._wall.zero_(); _e._fig8.zero_()
            _e._ring.zero_(); _e._ring_ecc.zero_(); _e._node_l.zero_(); _e._node_m.zero_()
            _e._node_k.zero_(); _e._node_w.zero_(); _e._node_v.zero_(); _e._tide.zero_()

        while not self.closed:
            t0 = loop.time()                                  # frame start, for steady pacing
            players = list(self.players.values())
            for p in players:
                if p.ctrl["spin"] is not None:                # Q/E -> orbit direction
                    p.spin_sign = p.ctrl["spin"]; p.ctrl["spin"] = None
                if p.ctrl["map"] is not None:
                    self.map_choice = p.ctrl["map"] if p.ctrl["map"] >= 0 else None
                    p.ctrl["map"] = None
                if p.ctrl["reset"]:
                    self.reset_flag = True; p.ctrl["reset"] = False
            if self.reset_flag:
                session.engine._map_choice = self.map_choice   # picked map (None=random)
                session.reset(); self.reset_flag = False; hold = 0; logged = False
                clear_knobs(session.engine)
            elif session.done:
                hold += 1
                if hold > TICK_HZ * 2.5:               # show the result ~2.5s, then new game
                    session.engine._map_choice = self.map_choice
                    session.reset(); hold = 0; logged = False
                    clear_knobs(session.engine)
            else:
                _e = session.engine
                clear_knobs(_e)
                base_cs = max(1, round(_e.W / 96))
                speeds = [base_cs] * _e.T
                humans = {}
                for p in players:
                    if p.ctrl["dir"] and (p.ctrl["dir"][0] or p.ctrl["dir"][1]):
                        p.last_dir = p.ctrl["dir"]            # heading the Drill/Wall point at
                    speeds[p.team] = _apply_player_stance(
                        _e, p.team, p.ctrl, p.spin_sign, p.last_dir, p.c0_hist, n)
                    humans[p.team] = (p.ctrl["target"], p.ctrl["dir"])
                _e._cursor_speed_t = speeds                   # per-seat (Doom slows ITS holder only)
                session.step(humans)
            blob = session.frame_blob()
            st = session.state(); st["fps"] = round(fps, 1)
            st["players"] = len(players)
            st["seats"] = sorted(self.players)            # which teams are humans (lobby chips)
            for p in players:
                p.c0_hist.append(list(st["cursors"][p.team])); del p.c0_hist[:-7]   # comet aim window
            if session.done and not logged:
                w = max(range(len(st["fighters"])), key=lambda i: st["fighters"][i])
                print(f"[telemetry] GAME END room={self.key} map={st['map']} tick={st['tick']} winner=team{w} "
                      f"counts={st['fighters']} spread={st['spread']} fps={fps:.1f}", flush=True)
                logged = True
            elif not session.done and st["tick"] and st["tick"] % 150 == 0:
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
                p.offer((blob, msg))
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
    key = q.get("room") or f"~solo-{id(sock)}"
    room = ROOMS.get(key)
    if room is None or room.closed:
        room = Room(key, mode=q.get("mode", "play"),
                    opponent=q.get("opponent", "latest"),
                    teams=int(q.get("teams", "2")),
                    size=q.get("size", "full"))
        ROOMS[key] = room
    player = room.join(sock)
    if player is None:                                  # room full
        await sock.close()
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
                if s == 0 and ctrl["stance"] == 0:    # re-tap Swarm cycles cloud -> comet
                    ctrl["swarm_mode"] ^= 1
                if s == 7 and ctrl["stance"] == 7:    # re-tap Atom cycles orbital -> binary star
                    ctrl["atom_mode"] ^= 1
                ctrl["stance"] = s
            elif "dir" in msg:                 # keyboard (arrows/WASD)
                ctrl["dir"] = msg["dir"]
            elif "target" in msg:              # mouse
                ctrl["target"] = msg["target"]
    except WebSocketDisconnect:
        pass
    finally:
        room.leave(player.team)
        if room.closed:
            ROOMS.pop(key, None)


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


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    ck = _latest_checkpoint()
    return {"ok": True, "device": DEVICE, "tick_hz": TICK_HZ,
            "latest_ckpt": ck and Path(ck).name}


if _STATIC.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC), html=True), name="static")
