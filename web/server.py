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
import os
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from rl.eval import _heuristic_dydx
from rl.policy import CursorPolicy, act
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
    def _ai_dydx(self) -> torch.Tensor:
        """Actions for ALL teams from the configured AI (policy or heuristic)."""
        if self.policy is None:
            return _heuristic_dydx(self.engine)
        obs = self.engine.get_observation()
        dydx, _, _, _ = act(self.policy, obs, self.engine.T,
                            self.engine.team_alive, deterministic=True)
        return dydx

    @torch.no_grad()
    def step(self, human_target: list[int] | None,
             human_dir: list[int] | None = None) -> None:
        dydx = self._ai_dydx()
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
        return {
            "tick": int(e.tick),
            "h": e.H, "w": e.W, "teams": e.T,
            "opponent": self.ckpt_name,
            "grid_b64": base64.b64encode(cell.cpu().numpy().tobytes()).decode(),
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
        height=int(os.environ.get("LW_PLAY_H", "192")),     # finer grid + more,
        width=int(os.environ.get("LW_PLAY_W", "288")),      # smaller units -> a
        fighters=int(os.environ.get("LW_PLAY_FIGHTERS", "8000")),  # dense fluid mass
    )
    ctrl: dict[str, Any] = {"target": None, "dir": None, "alive": True,
                            "reset": False, "pulse": False, "map": None}  # map=None -> random

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
                elif msg.get("pulse"):             # spacebar -> Pulse surge
                    ctrl["pulse"] = True
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
        PULSE_DUR, PULSE_CD, PULSE_MULT = 18, 180, 6.0   # active ticks / cooldown ticks / dmg x
        n = 0
        pulse_start = -PULSE_CD                          # monotonic tick of the last Pulse
        next_dl = loop.time()                            # absolute frame deadline (drift-corrected)
        while ctrl["alive"]:
            t0 = loop.time()                                  # frame start, for steady pacing
            if ctrl["reset"]:
                session.engine._map_choice = ctrl["map"]   # apply the picked map (None=random)
                session.reset(); ctrl["reset"] = False; hold = 0; logged = False
                pulse_start = -PULSE_CD; session.engine._surge = None   # clear pulse per game
            elif session.done:
                hold += 1
                if hold > TICK_HZ * 2.5:               # show the result ~2.5s, then new game
                    session.engine._map_choice = ctrl["map"]   # keep the picked map across games
                    session.reset(); hold = 0; logged = False
                    pulse_start = -PULSE_CD; session.engine._surge = None
            else:
                if ctrl["pulse"]:                       # consume request; fire only if off cooldown
                    if n - pulse_start >= PULSE_CD:
                        pulse_start = n
                    ctrl["pulse"] = False
                if 0 <= n - pulse_start < PULSE_DUR:    # human team (0) surges
                    s = torch.ones(1, session.engine.T, device=session.engine.device)
                    s[0, 0] = PULSE_MULT
                    session.engine._surge = s
                else:
                    session.engine._surge = None
                session.step(ctrl["target"], ctrl["dir"])
            st = session.state(); st["fps"] = round(fps, 1)
            st["pulse_active"] = bool(0 <= n - pulse_start < PULSE_DUR)
            st["pulse_cd"] = round(min(1.0, (n - pulse_start) / PULSE_CD), 2)  # 1.0 = ready
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
