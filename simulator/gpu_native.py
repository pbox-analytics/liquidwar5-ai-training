"""
GPU-native Liquid War simulator — entire game runs on GPU.

One CUDA thread block per game. All game state in shared memory.
No Python round-trips during gameplay. Results returned after
all ticks complete.

Usage:
    from simulator.gpu_native import run_games_gpu

    # Run 1000 games, each for 5000 ticks
    results = run_games_gpu(batch_size=1000, num_teams=4, max_ticks=5000)
    print(f"Team 0 won {(results[:, 0] > results[:, 1:4].max(dim=1).values).sum()} games")
"""

import os
import time
import torch
from torch.utils.cpp_extension import load_inline

# Load the CUDA source
_CUDA_SRC_PATH = os.path.join(os.path.dirname(__file__), "cuda_engine.cu")

_module = None


def _get_module():
    """JIT compile the CUDA kernel on first use."""
    global _module
    if _module is not None:
        return _module

    import os as _os
    _os.environ.setdefault("CUDA_HOME", "/usr")
    _os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0+PTX")

    with open(_CUDA_SRC_PATH) as f:
        cuda_src = f.read()

    print("Compiling GPU-native engine (first time only)...")
    cpp_decl = """
    #include <torch/extension.h>
    torch::Tensor run_games(torch::Tensor walls, int num_teams,
                            int fighters_per_team, int max_ticks,
                            int grad_iters);
    """

    _module = load_inline(
        name="liquid_war_cuda",
        cpp_sources=[cpp_decl],
        cuda_sources=[cuda_src],
        functions=["run_games"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )
    print("Compilation complete.")
    return _module


def generate_walls(batch_size, density=0.12, device='cuda'):
    """Generate random 64x64 maps with walls."""
    walls = torch.zeros(batch_size, 64, 64, dtype=torch.bool, device=device)
    walls[:, 0, :] = True
    walls[:, -1, :] = True
    walls[:, :, 0] = True
    walls[:, :, -1] = True
    walls[:, 1:-1, 1:-1] = torch.rand(
        batch_size, 62, 62, device=device) < density
    return walls


def run_games_gpu(batch_size=256, num_teams=4, fighters_per_team=500,
                  max_ticks=5000, grad_iters=4, wall_density=0.12,
                  walls=None, device='cuda'):
    """Run a batch of complete games on GPU.

    One CUDA thread block per game. All ticks execute on GPU
    without returning to Python.

    Args:
        batch_size: Number of games.
        num_teams: Teams per game (2-4).
        fighters_per_team: Starting fighters.
        max_ticks: Maximum ticks per game.
        grad_iters: Gradient spread iterations per tick.
        wall_density: Random wall probability.
        walls: Optional (B, 64, 64) bool tensor.
        device: CUDA device.

    Returns:
        Dict with results per game.
    """
    mod = _get_module()

    if walls is None:
        walls = generate_walls(batch_size, wall_density, device)

    start = time.time()
    # Returns (B, 5): [team0, team1, team2, team3, ticks]
    raw = mod.run_games(walls, num_teams, fighters_per_team,
                        max_ticks, grad_iters)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    fighters = raw[:, :num_teams]
    ticks = raw[:, -1].int()
    total = fighters.sum(dim=1)
    best = fighters.max(dim=1)

    return {
        'fighters_per_team': fighters,
        'ticks': ticks,
        'total_fighters': total,
        'best_team': best.indices,
        'best_count': best.values,
        'dominance': best.values / total.clamp(min=1),
        'elapsed': elapsed,
        'games_per_sec': batch_size / elapsed,
        'ticks_per_sec': (ticks.float().mean().item() * batch_size) / elapsed,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GPU-native Liquid War batch simulator")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--teams", type=int, default=4)
    parser.add_argument("--fighters", type=int, default=500)
    parser.add_argument("--ticks", type=int, default=5000)
    parser.add_argument("--grad-iters", type=int, default=4)
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"SMs: {torch.cuda.get_device_properties(0).multi_processor_count}")
    print(f"Running {args.batch_size} games, {args.ticks} ticks each...")
    print()

    results = run_games_gpu(
        batch_size=args.batch_size,
        num_teams=args.teams,
        fighters_per_team=args.fighters,
        max_ticks=args.ticks,
        grad_iters=args.grad_iters,
    )

    print(f"Completed {args.batch_size} games in {results['elapsed']:.2f}s")
    print(f"  {results['games_per_sec']:.1f} games/sec")
    print(f"  {results['ticks_per_sec']:.0f} game-ticks/sec")
    print(f"  Avg dominance: {results['dominance'].mean():.4f}")
    print(f"  Avg ticks: {results['ticks'].float().mean():.0f}")
    print(f"  Games with winner: {(results['dominance'] >= 0.99).sum().item()}/{args.batch_size}")

    # Scale test
    print()
    print("Scaling test:")
    for bs in [82, 256, 512, 1024, 2048]:
        r = run_games_gpu(batch_size=bs, num_teams=args.teams,
                          fighters_per_team=args.fighters,
                          max_ticks=args.ticks, grad_iters=args.grad_iters)
        print(f"  batch={bs:5d}  {r['elapsed']:5.2f}s  "
              f"{r['games_per_sec']:7.1f} games/s  "
              f"{r['ticks_per_sec']:>9.0f} ticks/s")
