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
        dydx, ai_stance = self._ai_dydx()
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
        if ai_stance is not None:                  # AI opponents (teams 1..) hold their own stances
            apply_stances(self.engine, ai_stance, dydx, team_start=1)
        self.engine.step(dydx)

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
                            "wall_orient": 0}   # 0 = horizontal bar, 1 = vertical bar (tap 4 to flip)

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
                _e = session.engine; _e._surge = None; _e._blackhole_pos = None                   # clear all stance knobs per game
                _e._spin.zero_(); _e._burst.zero_(); _e._drill.zero_(); _e._wall.zero_(); _e._fig8.zero_()
            elif session.done:
                hold += 1
                if hold > TICK_HZ * 2.5:               # show the result ~2.5s, then new game
                    session.engine._map_choice = ctrl["map"]   # keep the picked map across games
                    session.reset(); hold = 0; logged = False
                    _e = session.engine; _e._surge = None; _e._blackhole_pos = None
                    _e._spin.zero_(); _e._burst.zero_(); _e._drill.zero_(); _e._wall.zero_(); _e._fig8.zero_()
            else:
                if ctrl["dir"] and (ctrl["dir"][0] or ctrl["dir"][1]):
                    last_dir = ctrl["dir"]                  # heading the Drill/Wall point at
                _e = session.engine
                _e._spin.zero_(); _e._burst.zero_(); _e._drill.zero_(); _e._wall.zero_(); _e._fig8.zero_()
                _e._surge = None; _e._blackhole_pos = None
                stance = ctrl["stance"]                     # 0 Swarm 1 Spin 2 Drill 3 Wall 4 Pulse
                if stance == 0:                             # Swarm: loose, varied-radius orbits (electron cloud)
                    _e._spin[0, 0] = 0.5 * spin_sign
                    _e._burst[0, 0] = 0.15                  # slight loosen -> a diffuse orbiting cloud
                elif stance == 1:                           # Spin: tighten in + whirl FAST -> a compact vortex
                    _e._spin[0, 0] = 1.7 * spin_sign
                    _e._burst[0, 0] = -0.4                  # pull tight so it spins fast (Q/E sets direction)
                elif stance == 2:                           # Drill: 3 modes (slow/med/fast) — grind vs advance
                    m = ctrl["drill_mode"]                  # 0 slow / 1 med / 2 fast (tap 3 to rev)
                    DSPIN = (0.3, 0.7, 1.5); DSURGE = (1.0, 2.0, 4.0); DADV = (1.0, 0.62, 0.34)
                    sgn = spin_sign if spin_sign != 0 else 1   # the bit always spins
                    _e._spin[0, 0] = DSPIN[m] * sgn         # faster spin -> harder grind, but looser + slower
                    _e._drill[0, 0, 0] = float(last_dir[0]) * DADV[m]  # |drill| = advance speed
                    _e._drill[0, 0, 1] = float(last_dir[1]) * DADV[m]
                    if DSURGE[m] > 1.0:                     # the spinning front grinds (chews) harder
                        s = torch.ones(1, _e.T, device=_e.device); s[0, 0] = DSURGE[m]; _e._surge = s
                elif stance == 3:                           # Wall: a concentrated bar, horizontal or vertical (tap 4 to flip)
                    if ctrl["wall_orient"] == 0:            # horizontal bar = vertical facing
                        _e._wall[0, 0, 0] = 1.0; _e._wall[0, 0, 1] = 0.0
                    else:                                   # vertical bar = horizontal facing
                        _e._wall[0, 0, 0] = 0.0; _e._wall[0, 0, 1] = 1.0
                    _e._burst[0, 0] = -0.5                  # pull inward -> a tighter, denser, more concentrated bar
                elif stance == 4:                           # Pulse: concentric rings + damage waves
                    ring = math.sin(n * 0.33)
                    _e._burst[0, 0] = 1.0 if ring > 0 else -0.6
                    s = torch.ones(1, _e.T, device=_e.device)
                    if ring > 0.5:
                        s[0, 0] = 4.0                       # surge on each ring's crest
                    _e._surge = s
                elif stance == 5:                           # Doom: violent black-hole implosion
                    sgn = spin_sign if spin_sign != 0 else 1
                    _e._spin[0, 0] = 0.25 * sgn             # almost no swirl — pure radial collapse
                    _e._burst[0, 0] = -6.5                  # OVERWHELMING inward pull on its own mass
                    s = torch.ones(1, _e.T, device=_e.device); s[0, 0] = 6.0; _e._surge = s  # tidal devastation
                    _e._blackhole_pos = _e.cursor_pos[:, 0].float().clone()  # gravity well at YOUR cursor;
                    _e._blackhole_team = 0                                    # pull ∝ YOUR mass (real black hole):
                    _mass = float(_e.team_oh[0, 0].sum())                     # full army -> devastating well,
                    _frac = _mass / max(1.0, _e.fighters_per_team)
                    _e._blackhole_str = ctrl["doom_level"] * 55.0 * _frac ** 1.5  # STRONGER + super-linear in mass: full army -> devastating well, whittled -> fizzles fast (x1/x2/x3 charge, tap 6)
                    _e._blackhole_range = max(40.0, _e.W * 0.30)             # reach scales with the map (~30% of width) so it actually reaches + strips the enemy, not a tiny fixed radius
                elif stance == 6:                           # Maelstrom: fast wide orbiting shell (whirlpool)
                    sgn = spin_sign if spin_sign != 0 else 1
                    _e._spin[0, 0] = 2.0 * sgn              # whirl fast and wide around the cursor
                    _e._burst[0, 0] = 0.6                   # pushed out into a spinning ring, not a dense core
                elif stance == 7:                           # Atom: figure-8 electron orbitals
                    sgn = spin_sign if spin_sign != 0 else 1
                    _e._spin[0, 0] = 1.8 * sgn              # orbital speed
                    _e._burst[0, 0] = 0.4                   # a little room so the two lobes form
                    _e._fig8[0, 0] = 1.0                    # flip orbit across the cursor -> figure-8 loops
                # Doom charge trades speed for pull: at level L the cursor only moves
                # every L-th tick, so a 3x well is sluggish to reposition.
                _slow = ctrl["stance"] == 5 and ctrl["doom_level"] > 1 and (n % ctrl["doom_level"]) != 0
                session.step(None if _slow else ctrl["target"], [0, 0] if _slow else ctrl["dir"])
            st = session.state(); st["fps"] = round(fps, 1)
            st["stance"] = STANCES[ctrl["stance"]]      # held tactical state
            st["mode"] = (("slow", "med", "fast")[ctrl["drill_mode"]] if ctrl["stance"] == 2
                          else f"{ctrl['doom_level']}x" if ctrl["stance"] == 5
                          else ("horiz", "vert")[ctrl["wall_orient"]] if ctrl["stance"] == 3 else "")
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
