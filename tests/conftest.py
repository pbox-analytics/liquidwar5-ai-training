"""Shared pytest fixtures for the Liquid War engine + RL policy suite.

Everything here is CPU-only and tiny so the whole suite stays well under ~30s.
Run with::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
        uv run --with pytest --with numpy python -m pytest tests/ -q
"""
from __future__ import annotations

import os
import pathlib

import pytest
import torch

# Force CPU regardless of how the harness was invoked. The engine has CUDA-graph
# fast paths but they are gated on ``self.gradient.is_cuda`` / ``B == 1`` and the
# opt-in ``_cuda_graph`` flag, so a CPU device never touches them.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

#: Repo root (this file lives in <root>/tests/).
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: web/server.py is parsed as TEXT (never imported — it pulls fastapi + torch and
#: constructs models at import time). Parity tests regex specific constants out.
SERVER_PY = REPO_ROOT / "web" / "server.py"


@pytest.fixture(scope="session", autouse=True)
def _deterministic_seed():
    """Seed RNGs once so a flaky random-action run can be reproduced."""
    torch.manual_seed(1234)
    import random

    random.seed(1234)


def make_engine(batch_size=2, height=20, width=28, num_teams=3,
                fighters_per_team=30, grad_iters=4, **kw):
    """Construct a small CPU engine (NOT reset — caller resets, optionally with
    a hand-built wall grid).

    Kept deliberately tiny: a few hundred ticks at these sizes runs in a couple
    of seconds on CPU.
    """
    from simulator.engine import LiquidWarEngine

    return LiquidWarEngine(
        batch_size=batch_size, height=height, width=width,
        num_teams=num_teams, fighters_per_team=fighters_per_team,
        device="cpu", grad_iters=grad_iters, **kw,
    )


@pytest.fixture
def engine_factory():
    """Return the :func:`make_engine` builder (so a test can pick its own size)."""
    return make_engine


@pytest.fixture
def small_engine():
    """A reset 2-batch, 3-team CPU engine ready to ``step``."""
    e = make_engine()
    e.reset()
    return e


@pytest.fixture(scope="session")
def server_src():
    """The text of web/server.py (parsed, never imported)."""
    assert SERVER_PY.is_file(), f"server.py not found at {SERVER_PY}"
    return SERVER_PY.read_text()
