#!/usr/bin/env python3
"""Front-end-agnostic game server for playing/spectating the trained policy.

The Python GPU engine is the single source of truth: this serves ONE live game
(batch_size=1, runs real-time on CPU) over a websocket, so any client — a Canvas
page now, a richer Pixi/native app later — just renders the streamed state and
sends cursor input. No fidelity gap, no C-engine bridge: you play the exact
engine the policy trained in.

Modes (websocket query param ``mode``):
  - ``play``     : you drive team 0 (mouse target -> cursor), the policy drives
                   the rest. Opponent = ``opponent`` query param
                   ('heuristic' | 'latest' | a checkpoint path).
  - ``spectate`` : all teams AI (policy and/or heuristic) — also how progress
                   videos get recorded.

Protocol: client sends ``{"target": [y, x]}`` (cell the human points at) or
``{"reset": true}``; server steps one tick and replies with a state frame
(walls + per-cell team ownership + cursors + fighter counts). Client-paced: one
step per message, so the client's frame rate is the tick rate.

Run: ``uvicorn web.server:app --host 0.0.0.0 --port 8080``
(deps: fastapi, uvicorn[standard]).
"""
from __future__ import annotations

import base64
import glob
import os
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from rl.eval import _heuristic_dydx
from rl.policy import CursorPolicy, act
from simulator.engine import LiquidWarEngine

CKPT_DIR = os.environ.get("LW_CKPT_DIR", "/opt/training/results")  # NFS mount
DEVICE = os.environ.get("LW_PLAY_DEVICE", "cpu")                   # 1 game: CPU is real-time
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

    def __init__(self, mode: str, opponent: str, teams: int = 4,
                 height: int = 80, width: int = 110, fighters: int = 500) -> None:
        self.mode = mode
        self.engine = LiquidWarEngine(batch_size=1, height=height, width=width,
                                      num_teams=teams, fighters_per_team=fighters,
                                      device=DEVICE)
        self.engine.reset()
        # Resolve the opponent policy once (None => heuristic).
        self.policy: CursorPolicy | None = None
        self.ckpt_name = opponent
        if opponent not in ("heuristic", "random"):
            path = _latest_checkpoint() if opponent == "latest" else opponent
            if path:
                self.policy = _load_policy(path)
                self.ckpt_name = Path(path).name

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
    def step(self, human_target: list[int] | None) -> None:
        dydx = self._ai_dydx()
        if self.mode == "play" and human_target is not None:
            # Human drives team 0: step its cursor toward the pointed-at cell.
            cy, cx = self.engine.cursor_pos[0, 0].tolist()
            ty, tx = human_target
            dydx[0, 0, 0] = max(-1, min(1, ty - cy))
            dydx[0, 0, 1] = max(-1, min(1, tx - cx))
        self.engine.step(dydx)

    def reset(self) -> None:
        self.engine.reset()

    def state(self) -> dict[str, Any]:
        e = self.engine
        oh = e.team_oh[0]                                  # (T,H,W) presence
        present = oh.sum(0) > 0
        cell = oh.argmax(0).to(torch.int8)                 # team idx where present
        cell = torch.where(present, cell, torch.full_like(cell, -1))
        cell = torch.where(e.walls[0], torch.full_like(cell, -2), cell)  # -2 = wall
        return {
            "tick": int(e.tick),
            "h": e.H, "w": e.W, "teams": e.T,
            "opponent": self.ckpt_name,
            "grid_b64": base64.b64encode(cell.cpu().numpy().tobytes()).decode(),
            "cursors": e.cursor_pos[0].tolist(),           # [[y,x], ...] per team
            "fighters": oh.sum(dim=(1, 2)).long().tolist(),
            "alive": e.team_alive[0].tolist(),
            "done": bool(e.team_alive[0].sum() <= 1),
        }


app = FastAPI(title="liquidwar play")


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    await sock.accept()
    q = sock.query_params
    session = GameSession(
        mode=q.get("mode", "play"),
        opponent=q.get("opponent", "latest"),
        teams=int(q.get("teams", "4")),
    )
    await sock.send_json(session.state())
    try:
        while True:
            msg = await sock.receive_json()
            if msg.get("reset"):
                session.reset()
            else:
                session.step(msg.get("target"))
            await sock.send_json(session.state())
    except WebSocketDisconnect:
        return


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    ck = _latest_checkpoint()
    return {"ok": True, "device": DEVICE, "latest_ckpt": ck and Path(ck).name}


if _STATIC.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC), html=True), name="static")
