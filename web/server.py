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
from rl.policy import CursorPolicy, act, apply_stances
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


def _load_policy(path: str) -> CursorPolicy:
    policy = CursorPolicy().to(DEVICE)
    policy.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    policy.eval()
    return policy


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
        self.engine._surge = torch.ones(1, teams, device=DEVICE)  # per-team damage mult; IN-PLACE writes only (graph input)
        self.engine._wells_enabled = True            # play casts real cross-team wells (slots in engine.reset)
        # CUDA-graph capture/replay of the engine tick (the play loop is
        # launch-overhead-bound at B=1). Kill switch: LW_CUDA_GRAPH=0.
        self.engine._cuda_graph = os.environ.get("LW_CUDA_GRAPH", "1") != "0"
        self.engine.reset()
        self.policy: CursorPolicy | None = None
        self.ckpt_name = opponent
        if opponent not in ("heuristic", "random"):
            path = _latest_checkpoint() if opponent == "latest" else opponent
            if path:
                self.policy = _load_policy(path)
                self.ckpt_name = Path(path).name

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
        return dydx, stance

    @torch.no_grad()
    def step(self, human_target: list[int] | None,
             human_dir: list[int] | None = None) -> None:
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
            if human_dir is not None and (human_dir[0] or human_dir[1]):
                # Keyboard (arrows/WASD): drive the cursor directly, LW-style.
                dydx[0, 0, 0] = max(-1, min(1, human_dir[0]))
                dydx[0, 0, 1] = max(-1, min(1, human_dir[1]))
            elif human_target is not None:
                # Mouse: cursor homes toward the pointed-at cell.
                cy, cx = self.engine.cursor_pos[0, 0].tolist()
                ty, tx = human_target
                dydx[0, 0, 0] = max(-1, min(1, ty - cy))
                dydx[0, 0, 1] = max(-1, min(1, tx - cx))
            else:
                # No human input: team 0 stays put — never let the AI drive YOUR
                # cursor (that read as "the cursor moves on its own").
                dydx[0, 0, 0] = 0
                dydx[0, 0, 1] = 0
        # Re-cast the AI wells fresh each tick: zero the AI slots (1..) — the
        # human's slot 0 was just written by the ws loop. All writes in-place
        # (the slots are graph input buffers).
        e = self.engine
        e._doom_str[:, 1:] = 0; e._doom_horizon[:, 1:] = 0; e._doom_cap[:, 1:] = 0
        e._vortex_str[:, 1:] = 0
        self._ai_doom = None
        if ai_stance is not None:                  # AI opponents (teams 1..) hold their own stances
            apply_stances(self.engine, ai_stance, dydx, team_start=1)
            self._cast_ai_wells(ai_stance)
        self.engine.step(dydx)

    def _cast_ai_wells(self, ai_stance: torch.Tensor) -> None:
        """Cross-team PARITY for the AI: an opponent holding Doom (5) or
        Maelstrom (6) casts the same gravity well / whirlpool current at ITS
        cursor that the human gets — identical mass-scaled dials (Doom at 1x
        charge, Maelstrom undertow). Without this the AI's two weapon stances
        are cosmetic self-shapes and the human duels with superpowers the
        opponent lacks. ``_ai_doom`` feeds the client's hole shader."""
        e = self.engine
        stances = ai_stance[0].tolist()
        alive = e.team_alive[0].tolist()
        for t in range(1, e.T):
            s = stances[t]
            if s not in (5, 6) or not alive[t]:
                continue
            mass = float(e.team_oh[0, t].sum())
            frac = mass / max(1.0, e.fighters_per_team)
            blob_r = (mass / 3.14159) ** 0.5
            if s == 5:
                ring = 0.5 * (6.7 + (6.7 ** 2 + mass / 3.14159) ** 0.5)
                e._doom_pos[0, t] = e.cursor_pos[0, t].float()
                e._doom_str[0, t] = 32.0 * frac ** 1.5
                e._doom_range[0, t] = max(70.0, 2.2 * ring)
                e._doom_horizon[0, t] = max(14.0, 0.9 * blob_r)
                e._doom_cap[0, t] = 0.12
                if self._ai_doom is None:
                    self._ai_doom = [t, *e.cursor_pos[0, t].tolist()]
            else:
                e._vortex_pos[0, t] = e.cursor_pos[0, t].float()
                e._vortex_str[0, t] = 22.0 * frac ** 0.5
                e._vortex_range[0, t] = max(60.0, 1.5 * blob_r)
                e._vortex_sign[0, t] = 1.0
                e._vortex_rad[0, t] = 0.30

    def reset(self) -> None:
        self.engine.reset()

    def state(self) -> dict[str, Any]:
        e = self.engine
        oh = e.team_oh[0]                                  # (T,H,W) presence
        present = oh.sum(0) > 0
        cell = oh.argmax(0).to(torch.int8)
        cell = torch.where(present, cell, torch.full_like(cell, -1))
        cell = torch.where(e.walls[0], torch.full_like(cell, -2), cell)  # -2 = wall
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
        # Particle stream for the animated client: a fixed-stride sample of fighter
        # positions (identity-stable indices) + their team. The client interpolates
        # each mote between frames and streaks it along its velocity (curr - prev),
        # so motion is smooth 60fps flow rather than blinking cells. int16 positions
        # pack tighter than the per-cell grid.
        step = max(1, e.N // 9000)
        pidx = torch.arange(0, e.N, step, device=e.device)
        pos = torch.stack((e.fy[0, pidx], e.fx[0, pidx]), dim=1).reshape(-1).to(torch.int16)
        pteam = e.fteam[0, pidx].to(torch.uint8)
        return {
            "tick": int(e.tick),
            "h": e.H, "w": e.W, "teams": e.T,
            "opponent": self.ckpt_name,
            "grid_b64": base64.b64encode(cell.cpu().numpy().tobytes()).decode(),
            "pos_b64": base64.b64encode(pos.cpu().numpy().tobytes()).decode(),
            "pteam_b64": base64.b64encode(pteam.cpu().numpy().tobytes()).decode(),
            "pn": int(pidx.numel()),
            "cursors": e.cursor_pos[0].tolist(),
            "alive": e.team_alive[0].tolist(),
            "ai_doom": getattr(self, "_ai_doom", None),   # [team, y, x] while an AI holds Doom

            "done": self.done,
            "map": MAP_NAMES[e._last_arch] if 0 <= getattr(e, "_last_arch", -1) < len(MAP_NAMES) else "?",
            **self._hud,
        }


app = FastAPI(title="liquidwar play")


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    """Server-driven game loop + async input — steady frame rate, no round-trip jitter."""
    await sock.accept()
    q = sock.query_params
    session = GameSession(
        mode=q.get("mode", "play"),
        opponent=q.get("opponent", "latest"),
        teams=int(q.get("teams", "2")),
        height=int(os.environ.get("LW_PLAY_H", "384")),     # doubled-again battlefield
        width=int(os.environ.get("LW_PLAY_W", "576")),      # (~4x original area; ~50fps at 8000)
        fighters=int(os.environ.get("LW_PLAY_FIGHTERS", "8000")),
    )
    ctrl: dict[str, Any] = {"target": None, "dir": None, "alive": True, "reset": False,
                            "map": None, "spin": None, "stance": 0, "drill_mode": 0, "doom_level": 1,
                            "wall_orient": 0,   # 0 = horizontal bar, 1 = vertical bar (tap 4 to flip)
                            "pulse_mode": 0,    # 0 wave / 1 cymatic rings / 2 cymatic star (tap 5 to cycle)
                            "spin_mode": 0,     # 0 calm / 1 vortex / 2 frenzy (tap 2 to shift gear)
                            "mael_mode": 0}     # 0 undertow / 1 ejecta / 2 shear (tap 7 to cycle)

    async def receiver() -> None:
        try:
            while True:
                msg = await sock.receive_json()
                if msg.get("reset"):
                    ctrl["reset"] = True
                elif "map" in msg:                 # map picker -> force archetype + new game
                    m = msg["map"]
                    ctrl["map"] = None if m is None or m < 0 else int(m)
                    ctrl["reset"] = True
                elif "spin" in msg:                # Q/E -> orbit direction (Spin/Swarm stances)
                    ctrl["spin"] = float(msg["spin"])
                elif "stance" in msg:              # 1-5 -> select held stance; re-tap Drill(3) revs its mode
                    s = max(0, min(7, int(msg["stance"])))
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
                    if s == 4 and ctrl["stance"] == 4:    # re-tap Pulse cycles wave -> rings -> star
                        ctrl["pulse_mode"] = (ctrl["pulse_mode"] + 1) % 3
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
                    ctrl["stance"] = s
                elif "dir" in msg:                 # keyboard (arrows/WASD)
                    ctrl["dir"] = msg["dir"]
                elif "target" in msg:              # mouse
                    ctrl["target"] = msg["target"]
        except WebSocketDisconnect:
            ctrl["alive"] = False

    async def game_loop() -> None:
        dt = 1.0 / TICK_HZ
        hold = 0                                       # ticks to linger on a finished game
        fps = TICK_HZ                                  # achieved frame rate (EMA)
        prev = None
        logged = False                                 # one telemetry line per finished game
        loop = asyncio.get_event_loop()
        n = 0
        spin_sign = 1                                    # Q/E orbit direction (Spin/Swarm stances): +1/-1/0
        last_dir = [0, 1]                                # heading the Drill/Wall point at; default right
        next_dl = loop.time()                            # absolute frame deadline (drift-corrected)
        STANCES = ("Swarm", "Spin", "Drill", "Wall", "Pulse", "Doom", "Maelstrom", "Atom")  # index = ctrl["stance"]
        while ctrl["alive"]:
            t0 = loop.time()                                  # frame start, for steady pacing
            if ctrl["spin"] is not None:                      # Q/E -> orbit direction (Spin/Swarm stances)
                spin_sign = ctrl["spin"]; ctrl["spin"] = None
            if ctrl["reset"]:
                session.engine._map_choice = ctrl["map"]   # apply the picked map (None=random)
                session.reset(); ctrl["reset"] = False; hold = 0; logged = False
                _e = session.engine                                # clear all stance knobs per game
                _e._surge.fill_(1.0)   # neutral damage mult (in-place: graph input)
                _e._doom_str.zero_(); _e._doom_horizon.zero_(); _e._doom_cap.zero_(); _e._vortex_str.zero_()
                _e._spin.zero_(); _e._burst.zero_(); _e._drill.zero_(); _e._wall.zero_(); _e._fig8.zero_(); _e._ring.zero_(); _e._ring_ecc.zero_(); _e._node_l.zero_(); _e._node_m.zero_(); _e._node_k.zero_(); _e._node_w.zero_(); _e._node_v.zero_()
            elif session.done:
                hold += 1
                if hold > TICK_HZ * 2.5:               # show the result ~2.5s, then new game
                    session.engine._map_choice = ctrl["map"]   # keep the picked map across games
                    session.reset(); hold = 0; logged = False
                    _e = session.engine
                    _e._surge.fill_(1.0)   # neutral damage mult (in-place: graph input)
                    _e._doom_str.zero_(); _e._doom_horizon.zero_(); _e._doom_cap.zero_(); _e._vortex_str.zero_()
                    _e._spin.zero_(); _e._burst.zero_(); _e._drill.zero_(); _e._wall.zero_(); _e._fig8.zero_(); _e._ring.zero_(); _e._ring_ecc.zero_(); _e._node_l.zero_(); _e._node_m.zero_(); _e._node_k.zero_(); _e._node_w.zero_(); _e._node_v.zero_()
            else:
                if ctrl["dir"] and (ctrl["dir"][0] or ctrl["dir"][1]):
                    last_dir = ctrl["dir"]                  # heading the Drill/Wall point at
                _e = session.engine
                _e._spin.zero_(); _e._burst.zero_(); _e._drill.zero_(); _e._wall.zero_(); _e._fig8.zero_(); _e._ring.zero_(); _e._ring_ecc.zero_(); _e._node_l.zero_(); _e._node_m.zero_(); _e._node_k.zero_(); _e._node_w.zero_(); _e._node_v.zero_()
                _e._surge.fill_(1.0)   # neutral damage mult (in-place: graph input)
                _e._doom_str.zero_(); _e._doom_horizon.zero_(); _e._doom_cap.zero_(); _e._vortex_str.zero_()
                stance = ctrl["stance"]                     # 0 Swarm 1 Spin 2 Drill 3 Wall 4 Pulse
                if stance == 0:                             # Swarm: loose, varied-radius orbits (electron cloud)
                    _e._spin[0, 0] = 0.5 * spin_sign
                    _e._burst[0, 0] = 0.15                  # slight loosen -> a diffuse orbiting cloud
                elif stance == 1:                           # Spin: 3 forms (tap 2): vortex -> sawblade -> galaxy
                    sm = ctrl["spin_mode"]
                    sgn = spin_sign if spin_sign != 0 else 1
                    if sm == 0:                             # vortex: the classic compact fast spin
                        _e._spin[0, 0] = 1.7 * spin_sign
                        _e._burst[0, 0] = -0.4
                    elif sm == 1:                           # sawblade: dense disc + 8 rotating teeth
                        _e._spin[0, 0] = 1.6 * sgn
                        _e._burst[0, 0] = -0.45             # packed disc body
                        _e._node_m[0, 0] = 8.0              # 8 angular clusters = the teeth
                        _e._node_w[0, 0] = 0.4 * sgn        # the tooth pattern SWEEPS with the spin
                    else:                                   # galaxy: wide slow swirl with winding spiral arms
                        _e._spin[0, 0] = 1.1 * sgn
                        _e._burst[0, 0] = 0.35              # spread out -> a broad disc
                        _e._node_m[0, 0] = 3.0              # 3 arms
                        _e._node_k[0, 0] = 0.25 * sgn       # arms WIND with radius (logarithmic-spiral look)
                        _e._node_w[0, 0] = -0.05 * sgn      # slow majestic pattern rotation
                elif stance == 2:                           # Drill: 3 modes (slow/med/fast) — grind vs advance
                    m = ctrl["drill_mode"]                  # 0 slow / 1 med / 2 fast (tap 3 to rev)
                    DSPIN = (0.3, 0.7, 1.5); DSURGE = (1.0, 2.0, 4.0); DADV = (1.0, 0.62, 0.34)
                    sgn = spin_sign if spin_sign != 0 else 1   # the bit always spins
                    _e._spin[0, 0] = DSPIN[m] * sgn         # faster spin -> harder grind, but looser + slower
                    _e._drill[0, 0, 0] = float(last_dir[0]) * DADV[m]  # |drill| = advance speed
                    _e._drill[0, 0, 1] = float(last_dir[1]) * DADV[m]
                    if DSURGE[m] > 1.0:                     # the spinning front grinds (chews) harder
                        _e._surge[0, 0] = DSURGE[m]
                elif stance == 3:                           # Wall: a concentrated bar, horizontal or vertical (tap 4 to flip)
                    if ctrl["wall_orient"] == 0:            # horizontal bar = vertical facing
                        _e._wall[0, 0, 0] = 1.0; _e._wall[0, 0, 1] = 0.0
                    else:                                   # vertical bar = horizontal facing
                        _e._wall[0, 0, 0] = 0.0; _e._wall[0, 0, 1] = 1.0
                    _e._burst[0, 0] = -0.9                  # strong inward pull -> a dense solid COLUMN, not a picket line
                elif stance == 4:                           # Pulse: 3 modes (tap 5 to cycle)
                    pm = ctrl["pulse_mode"]
                    if pm == 0:                             # wave: traveling rings + damage crests (the original)
                        ring = math.sin(n * 0.33)
                        _e._burst[0, 0] = 1.0 if ring > 0 else -0.6
                        if ring > 0.5:
                            _e._surge[0, 0] = 4.0           # surge on each ring's crest
                    elif pm == 1:                           # rings: cymatic STANDING rings (Chladni circular mode)
                        _e._node_l[0, 0] = 12.0             # nodal wavelength — concentric resonance bands (was 14: tighter)
                        _e._node_v[0, 0] = 0.05             # breathe 2.5x the old fixed 0.02 — visibly pumping
                        _e._surge[0, 0] = 3.0               # (was 2.0) the resonance hits harder
                    else:                                   # star: cymatic nodal-diameter mode — a 6-petal figure
                        _e._node_l[0, 0] = 16.0
                        _e._node_m[0, 0] = 6.0
                        _e._node_w[0, 0] = 0.05             # petal sweep you can actually see (was 0.01 drift)
                        _e._node_v[0, 0] = 0.05             # rings pump under the petals
                        _e._surge[0, 0] = 3.0               # (was 2.0)
                elif stance == 5:                           # Doom: violent black-hole implosion
                    sgn = spin_sign if spin_sign != 0 else 1
                    lvl = ctrl["doom_level"]                # 1x/2x/3x charge (tap 6)
                    # The ARMY is the visual cue: it forms a spinning ACCRETION
                    # DISK — `_ring` holds an open orbit radius (the black hole's
                    # dark centre), the swirl spins it, both growing with charge.
                    # The client's lensing/amber-disk shader sits in the hole.
                    _e._spin[0, 0] = (1.2, 1.8, 2.4)[lvl - 1] * sgn
                    # tidal surge scales with charge (was a flat 6x at every level)
                    _e._surge[0, 0] = (4.0, 5.0, 6.0)[lvl - 1]
                    _e._doom_pos[0, 0] = _e.cursor_pos[0, 0].float()         # gravity well at YOUR cursor;
                    _mass = float(_e.team_oh[0, 0].sum())                     # pull ∝ YOUR mass (real black hole)
                    # Disk target radius is MASS-SCALED: fighters pack one-per-cell,
                    # so a fixed small radius just saturates back into a solid blob.
                    # Solve pi*(r_out^2 - r_in^2) = mass for the band centred on the
                    # target with its INNER edge at the rendered horizon (r_in,
                    # matching the client's 4.4+2.3*lvl graphic) -> the hole stays
                    # open and the whole army becomes the spinning accretion disk.
                    _r_in = (6.7, 9.0, 11.3)[lvl - 1]
                    _ring_val = 0.5 * (_r_in + (_r_in ** 2 + _mass / 3.14159) ** 0.5)
                    _e._ring[0, 0] = _ring_val
                    _e._ring_ecc[0, 0] = 1.0                # full oblate -> the edge-on Gargantua blade
                    _frac = _mass / max(1.0, _e.fighters_per_team)
                    _e._doom_str[0, 0] = lvl * 32.0 * _frac ** 1.5  # super-linear in mass: gentler pull peels the loosely-bound periphery off the enemy, not the whole army (x1/x2/x3 charge, tap 6)
                    # FINITE reach (was the full map diagonal, which made Doom
                    # inescapable -> an auto-win): ~2.2x the disk radius, so a
                    # dispersed or kiting enemy escapes the pull and Doom is a
                    # committed finisher, not a vacuum.
                    _e._doom_range[0, 0] = max(70.0, 2.2 * _ring_val)
                    # horizon ~ the disk's inner mass edge, not the whole blob
                    # radius (1.5x blob swallowed anything within ~75 cells)
                    _e._doom_horizon[0, 0] = max(14.0, 0.9 * (_mass / 3.14159) ** 0.5)
                    _e._doom_cap[0, 0] = 0.12                               # fraction devoured per tick (was 0.18)
                elif stance == 6:                           # Maelstrom: a whirlpool CURRENT (cross-team vorticity)
                    sgn = spin_sign if spin_sign != 0 else 1
                    mm = ctrl["mael_mode"]                  # 0 undertow / 1 ejecta / 2 shear (tap 7)
                    # Your army IS the whirlpool body: the ORIGINAL big loose
                    # form — a fast wide orbiting shell, burst pushing the mass
                    # out into a broad swirling storm-cloud (no ring, no packed
                    # disk; nothing like Doom's silhouette).
                    _e._spin[0, 0] = 2.0 * sgn
                    _e._burst[0, 0] = 0.6
                    # The CURRENT — Doom's radial devour rotated 90°: enemies
                    # near the well are swept TANGENTIALLY off their gradient
                    # and entrained into orbit through your storm-cloud; no
                    # capture, attrition by ordinary contact combat. Strength
                    # scales with YOUR mass — a whittled army stirs a weak eddy.
                    _mass = float(_e.team_oh[0, 0].sum())
                    _frac = _mass / max(1.0, _e.fighters_per_team)
                    _e._vortex_pos[0, 0] = _e.cursor_pos[0, 0].float()
                    _e._vortex_sign[0, 0] = float(sgn)      # current direction follows Q/E
                    # Server-tunable dial (cf. _doom_str). Nerfed from 30/2.0x
                    # after play: with the flat falloff it yanked units off the
                    # enemy blob from across the arena. Now 22 + 1.5x-blob reach
                    # + a SQUARED falloff in the engine = a local current you
                    # can see coming and skirt, still lethal to wade through.
                    _e._vortex_str[0, 0] = 22.0 * _frac ** 0.5
                    _e._vortex_range[0, 0] = max(60.0, 1.5 * (_mass / 3.14159) ** 0.5)
                    # radial component per mode: undertow spirals them inward to
                    # the rim, ejecta flings entrained enemies outward (scatters
                    # a formation off its cursor), shear is pure deflection
                    _e._vortex_rad[0, 0] = (0.30, -0.45, 0.0)[mm]
                elif stance == 7:                           # Atom: figure-8 electron orbitals
                    sgn = spin_sign if spin_sign != 0 else 1
                    _e._spin[0, 0] = 1.8 * sgn              # orbital speed
                    _e._burst[0, 0] = 0.4                   # a little room so the two lobes form
                    _e._fig8[0, 0] = 1.0                    # flip orbit across the cursor -> figure-8 loops
                # Doom charge trades speed for pull: at level L the cursor moves at 1/L
                # cells-per-tick — every tick, so it's a smooth slow glide, not a stutter —
                # making a 3x well sluggish to reposition.
                _base_cs = max(1, round(_e.W / 96))
                _e.cursor_speed = (max(1, _base_cs // ctrl["doom_level"])
                                   if ctrl["stance"] == 5 else _base_cs)
                session.step(ctrl["target"], ctrl["dir"])
            st = session.state(); st["fps"] = round(fps, 1)
            st["stance"] = STANCES[ctrl["stance"]]      # held tactical state
            st["mode"] = (("slow", "med", "fast")[ctrl["drill_mode"]] if ctrl["stance"] == 2
                          else f"{ctrl['doom_level']}x" if ctrl["stance"] == 5
                          else ("horiz", "vert")[ctrl["wall_orient"]] if ctrl["stance"] == 3
                          else ("wave", "rings", "star")[ctrl["pulse_mode"]] if ctrl["stance"] == 4
                          else ("vortex", "sawblade", "galaxy")[ctrl["spin_mode"]] if ctrl["stance"] == 1
                          else ("undertow", "ejecta", "shear")[ctrl["mael_mode"]] if ctrl["stance"] == 6 else "")
            st["spin_dir"] = spin_sign
            if session.done and not logged:
                w = max(range(len(st["fighters"])), key=lambda i: st["fighters"][i])
                print(f"[telemetry] GAME END map={st['map']} tick={st['tick']} winner=team{w} "
                      f"counts={st['fighters']} spread={st['spread']} fps={fps:.1f}", flush=True)
                logged = True
            elif not session.done and st["tick"] and st["tick"] % 150 == 0:
                print(f"[telemetry] tick={st['tick']} fps={fps:.1f} flood={st['flood_pct']}% "
                      f"spread={st['spread']} counts={st['fighters']}", flush=True)
            try:
                await sock.send_json(st)
            except Exception:
                break
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
                print(f"[perf] work={(t_work - t0) * 1000:.1f}ms period={(now - t0) * 1000:.1f}ms fps={fps:.0f}", flush=True)
            prev = now
            n += 1                                     # advance the pulse clock (was missing -> Pulse stuck on)

    await asyncio.gather(receiver(), game_loop())


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    ck = _latest_checkpoint()
    return {"ok": True, "device": DEVICE, "tick_hz": TICK_HZ,
            "latest_ckpt": ck and Path(ck).name}


if _STATIC.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC), html=True), name="static")
