#!/usr/bin/env python3
"""Win-rate evaluation for the Liquid War cursor policy.

In self-play PPO the training return is deceptive: both sides improve together,
so the dominance-delta reward hovers near zero even as the policy gets much
better. To actually tell whether a checkpoint is learning, play it against a
FIXED opponent and measure win-rate:

  - ``random``     : sanity floor (a real policy should crush this).
  - ``heuristic``  : the built-in toward-enemy-centroid AI (the same logic as
                     ``LiquidWarEngine.step_with_ai``) = the meaningful benchmark,
                     "does the net beat the tuned heuristic?".
  - a ``.pt`` path : a frozen earlier checkpoint (self-play progress / Elo).

The eval policy drives team 0; the opponent drives teams 1..T-1; the game runs
headless to elimination; team 0's win fraction over ``--games`` parallel games is
the win-rate. Run it across a run's ``upd_*.pt`` checkpoints to get the learning
curve — the thing the training ``ret`` cannot show you.

Notes
-----
- Default ``--teams 4`` matches the training config (1 policy team vs 3 opponent
  teams in a free-for-all; >0.25 means better than an average team). Use
  ``--teams 2`` for a clean head-to-head 1v1 vs the opponent (>0.5 = beats it).
- Eval is ``@torch.no_grad`` and greedy (deterministic actions).

Usage
-----
    python -m rl.eval --ckpt results/rl/<run>/best.pt --opponent heuristic
    python -m rl.eval --ckpt-dir results/rl/<run> --opponent heuristic   # curve
    python -m rl.eval --ckpt a.pt --opponent b.pt --teams 2              # 1v1
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import torch

from rl.policy import CursorPolicy, act, apply_stances
from simulator.engine import LiquidWarEngine


@torch.no_grad()
def _heuristic_dydx(engine: LiquidWarEngine) -> torch.Tensor:
    """Per-team cursor move toward the enemy centroid.

    Replicates ``LiquidWarEngine.step_with_ai``'s inline AI as a pure action so
    it can drive a subset of teams (the opponents) while the policy drives the
    rest. ``engine.step`` handles the actual cursor movement + passability.

    :param engine: the live engine (reads ``health``, ``team_oh``, ``cursor_pos``).
    :returns: ``(B, T, 2)`` long dy/dx in ``{-1, 0, 1}``.
    """
    B, T, H, W = engine.B, engine.T, engine.H, engine.W
    has_fighter = (engine.health > 0).float()
    y_c = engine._y_idx.float().expand(B, H, W)
    x_c = engine._x_idx.float().expand(B, H, W)
    dydx = torch.zeros(B, T, 2, dtype=torch.long, device=engine.device)
    for t in range(T):
        enemy = (has_fighter - engine.team_oh[:, t] * has_fighter).clamp(min=0)
        e_count = enemy.sum(dim=(1, 2)).clamp(min=1)
        e_y = (enemy * y_c).sum(dim=(1, 2)) / e_count
        e_x = (enemy * x_c).sum(dim=(1, 2)) / e_count
        cy = engine.cursor_pos[:, t, 0].float()
        cx = engine.cursor_pos[:, t, 1].float()
        dydx[:, t, 0] = (e_y - cy).sign().long()
        dydx[:, t, 1] = (e_x - cx).sign().long()
    return dydx


@torch.no_grad()
def _opponent_dydx(opponent, engine: LiquidWarEngine, obs: torch.Tensor) -> torch.Tensor:
    """Actions for ALL teams from the opponent (only teams 1..T-1 are used).

    :param opponent: ``"heuristic"``, ``"random"``, or a :class:`CursorPolicy`.
    """
    B, T = engine.B, engine.T
    if opponent == "heuristic":
        return _heuristic_dydx(engine)
    if opponent == "random":
        return torch.randint(-1, 2, (B, T, 2), device=engine.device)
    dydx, _, _, _, _ = act(opponent, obs, T, engine.team_alive, deterministic=True)
    return dydx


@torch.no_grad()
def win_rate(eval_policy: CursorPolicy, opponent, *, games: int = 128,
             teams: int = 4, height: int = 80, width: int = 110,
             fighters: int = 500, device: str = "cuda",
             max_ticks: int = 3000) -> float:
    """Fraction of ``games`` that team 0 (``eval_policy``) wins vs ``opponent``.

    :param eval_policy: the policy under evaluation (drives team 0).
    :param opponent: ``"heuristic"`` | ``"random"`` | a :class:`CursorPolicy`
        (drives teams 1..T-1).
    :returns: team-0 win fraction in ``[0, 1]``.
    """
    engine = LiquidWarEngine(batch_size=games, height=height, width=width,
                             num_teams=teams, fighters_per_team=fighters,
                             device=device)
    engine.reset()
    B, T = engine.B, engine.T
    winners = torch.full((B,), -1, dtype=torch.long, device=device)  # -1 = unfinished
    for _ in range(max_ticks):
        obs = engine.get_observation()
        eval_dydx, eval_stance, _, _, _ = act(eval_policy, obs, T, engine.team_alive,
                                              deterministic=True)
        dydx = _opponent_dydx(opponent, engine, obs)
        dydx[:, 0] = eval_dydx[:, 0]                  # team 0 = eval policy
        # team 0 holds its chosen stance; opponents (1..) stay un-stanced (default).
        st = torch.zeros(B, T, dtype=torch.long, device=device)
        st[:, 0] = eval_stance[:, 0]
        apply_stances(engine, st, dydx)
        engine._spin[:, 1:] = 1.0; engine._burst[:, 1:] = 0.0
        engine._drill[:, 1:] = 0.0; engine._wall[:, 1:] = 0.0; engine._surge[:, 1:] = 1.0
        _, done, _ = engine.step(dydx)
        newly = done & (winners < 0)
        if newly.any():
            w = engine.team_oh.sum(dim=(2, 3)).argmax(dim=1)
            winners[newly] = w[newly]
        if (winners >= 0).all():
            break
    unfinished = winners < 0                          # hit max_ticks: award leader
    if unfinished.any():
        w = engine.team_oh.sum(dim=(2, 3)).argmax(dim=1)
        winners[unfinished] = w[unfinished]
    return (winners == 0).float().mean().item()


def _load_policy(path: str, device: str) -> CursorPolicy:
    """Instantiate a :class:`CursorPolicy` (training defaults) and load weights."""
    policy = CursorPolicy().to(device)
    policy.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    policy.eval()
    return policy


def main() -> None:
    ap = argparse.ArgumentParser(description="Win-rate eval for the Liquid War cursor policy")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--ckpt", help="single checkpoint .pt to evaluate")
    grp.add_argument("--ckpt-dir", help="dir of upd_*.pt checkpoints -> a curve")
    ap.add_argument("--opponent", default="heuristic",
                    help="'heuristic' | 'random' | path to a frozen .pt opponent")
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--teams", type=int, default=4)
    ap.add_argument("--height", type=int, default=80)
    ap.add_argument("--width", type=int, default=110)
    ap.add_argument("--fighters", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    opponent = args.opponent
    if opponent not in ("heuristic", "random"):
        opponent = _load_policy(opponent, args.device)   # frozen-checkpoint opponent

    kw = dict(games=args.games, teams=args.teams, height=args.height,
              width=args.width, fighters=args.fighters, device=args.device)

    if args.ckpt:
        wr = win_rate(_load_policy(args.ckpt, args.device), opponent, **kw)
        print(f"{args.ckpt}  vs {args.opponent}:  win-rate = {wr:.3f}  "
              f"({args.games} games, {args.teams} teams)")
        return

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "upd_*.pt")))
    if not ckpts:
        ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.pt")))
    if not ckpts:
        raise SystemExit(f"no .pt checkpoints in {args.ckpt_dir}")
    print(f"# win-rate vs {args.opponent} ({args.games} games, {args.teams} teams)")
    print(f"{'checkpoint':<24} win_rate")
    for ckpt in ckpts:
        wr = win_rate(_load_policy(ckpt, args.device), opponent, **kw)
        print(f"{Path(ckpt).name:<24} {wr:.3f}", flush=True)


if __name__ == "__main__":
    main()
