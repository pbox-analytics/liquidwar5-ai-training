"""
Batch game runner using the PyTorch GPU engine.

Runs hundreds of games simultaneously on GPU, orders of magnitude
faster than the C binary approach.
"""

import time
import torch
from simulator.engine import LiquidWarEngine


def run_batch(batch_size=256, num_teams=4, max_ticks=24000,
              height=120, width=160, fighters_per_team=1500,
              device='cuda', verbose=True):
    """Run a batch of games to completion.

    Args:
        batch_size: Number of games to run simultaneously.
        num_teams: Teams per game (2-6).
        max_ticks: Maximum ticks per game.
        height, width: Map dimensions.
        fighters_per_team: Starting fighters per team.
        device: 'cuda' or 'cpu'.
        verbose: Print progress.

    Returns:
        Dict with per-game results.
    """
    engine = LiquidWarEngine(
        batch_size=batch_size,
        height=height,
        width=width,
        num_teams=num_teams,
        fighters_per_team=fighters_per_team,
        device=device,
    )

    state = engine.reset()

    start = time.time()
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Simple AI: move cursor toward center of enemy mass
    for tick in range(max_ticks):
        if done.all():
            break

        # AI: compute cursor actions
        actions = _simple_ai(engine)

        state, tick_done, info = engine.step(actions)
        done = done | tick_done

        if verbose and tick % 1000 == 0:
            elapsed = time.time() - start
            games_done = done.sum().item()
            dom = info['dominance'].mean().item()
            tps = (tick + 1) * batch_size / elapsed
            print(f"  tick {tick:>5}/{max_ticks}  "
                  f"done={games_done}/{batch_size}  "
                  f"avg_dom={dom:.3f}  "
                  f"ticks/sec={tps:.0f}")

    elapsed = time.time() - start
    info = engine._get_info()

    if verbose:
        total_ticks = engine.tick * batch_size
        print(f"\nCompleted {batch_size} games in {elapsed:.1f}s")
        print(f"  {total_ticks / elapsed:.0f} game-ticks/sec")
        print(f"  {batch_size / elapsed:.1f} games/sec")
        print(f"  Avg dominance: {info['dominance'].mean():.4f}")

    return {
        'fighters_per_team': info['fighters_per_team'].cpu(),
        'total_fighters': info['total_fighters'].cpu(),
        'best_team': info['best_team'].cpu(),
        'dominance': info['dominance'].cpu(),
        'ticks': torch.full((batch_size,), engine.tick),
        'elapsed_seconds': elapsed,
    }


def _simple_ai(engine):
    """Simple AI: move each cursor toward nearest enemy cluster.

    This is a placeholder — will be replaced by neural network.
    """
    B, T = engine.B, engine.T
    actions = torch.zeros(B, T, 2, dtype=torch.long, device=engine.device)

    for t in range(T):
        alive = engine.team_alive[:, t]
        if not alive.any():
            continue

        # Find centroid of enemy fighters
        enemy_mask = (engine.team_grid >= 0) & (engine.team_grid != t)
        if not enemy_mask.any():
            continue

        # Compute enemy centroid per game
        y_coords = torch.arange(engine.H, device=engine.device).float()
        x_coords = torch.arange(engine.W, device=engine.device).float()

        enemy_float = enemy_mask.float()
        enemy_count = enemy_float.sum(dim=(1, 2)).clamp(min=1)

        enemy_y = (enemy_float * y_coords.view(1, -1, 1)).sum(
            dim=(1, 2)) / enemy_count
        enemy_x = (enemy_float * x_coords.view(1, 1, -1)).sum(
            dim=(1, 2)) / enemy_count

        # Move cursor toward enemy centroid
        cy = engine.cursor_pos[:, t, 0].float()
        cx = engine.cursor_pos[:, t, 1].float()

        dy = (enemy_y - cy).sign().long()
        dx = (enemy_x - cx).sign().long()

        actions[:, t, 0] = torch.where(alive, dy, torch.zeros_like(dy))
        actions[:, t, 1] = torch.where(alive, dx, torch.zeros_like(dx))

    return actions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run batched Liquid War games on GPU")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--teams", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=12000)
    parser.add_argument("--height", type=int, default=100)
    parser.add_argument("--width", type=int, default=130)
    parser.add_argument("--fighters", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print(f"Running {args.batch_size} games on {args.device}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB")

    results = run_batch(
        batch_size=args.batch_size,
        num_teams=args.teams,
        max_ticks=args.ticks,
        height=args.height,
        width=args.width,
        fighters_per_team=args.fighters,
        device=args.device,
    )
