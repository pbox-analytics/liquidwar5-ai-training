#!/usr/bin/env python3
"""
Analyze evolution results from Liquid War 5 AI training.

Reads the output from evolve.py and displays:
- Best parameters found
- Fitness progression over generations
- Parameter distribution analysis

Usage:
    python3 analyze.py results/<timestamp>/
    python3 analyze.py results/<timestamp>/ --plot  # requires matplotlib
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def load_evolution_log(run_dir: Path) -> list:
    log_file = run_dir / "evolution_log.csv"
    if not log_file.exists():
        print(f"Error: {log_file} not found", file=sys.stderr)
        sys.exit(1)

    with open(log_file) as f:
        return list(csv.DictReader(f))


def load_best_params(run_dir: Path) -> dict:
    best_file = run_dir / "best_params.json"
    if not best_file.exists():
        return {}
    with open(best_file) as f:
        return json.load(f)


def print_summary(run_dir: Path):
    log = load_evolution_log(run_dir)
    best = load_best_params(run_dir)

    print(f"=== Evolution Run: {run_dir.name} ===\n")

    if best:
        print("Best Parameters Found:")
        print(f"  Fitness: {best['fitness']:.4f}")
        for k, v in best["params"].items():
            print(f"  {k}: {v}")
        print(f"\n  CLI args: {best['cli_args']}")
        print()

    print("Generation Progress:")
    print(f"  {'Gen':>4} {'Best':>8} {'Avg':>8} {'Time':>7}")
    print(f"  {'---':>4} {'----':>8} {'---':>8} {'----':>7}")

    for row in log:
        gen = int(row["generation"])
        best_f = float(row["best_fitness"])
        avg_f = float(row["avg_fitness"])
        t = float(row["time_seconds"])
        print(f"  {gen:4d} {best_f:8.4f} {avg_f:8.4f} {t:6.1f}s")

    if log:
        first_best = float(log[0]["best_fitness"])
        last_best = float(log[-1]["best_fitness"])
        improvement = last_best - first_best
        print(f"\n  Improvement: {first_best:.4f} -> {last_best:.4f} "
              f"({improvement:+.4f})")


def plot_evolution(run_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib",
              file=sys.stderr)
        sys.exit(1)

    log = load_evolution_log(run_dir)

    gens = [int(r["generation"]) for r in log]
    best_fitness = [float(r["best_fitness"]) for r in log]
    avg_fitness = [float(r["avg_fitness"]) for r in log]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Fitness over generations
    ax = axes[0]
    ax.plot(gens, best_fitness, "b-", label="Best fitness", linewidth=2)
    ax.plot(gens, avg_fitness, "r--", label="Avg fitness", alpha=0.7)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Key parameter evolution
    ax = axes[1]
    density_w = [int(r["best_density_weight"]) for r in log]
    replan = [int(r["best_replan"]) for r in log]
    candidates = [int(r["best_candidates"]) for r in log]

    ax.plot(gens, density_w, "g-", label="density_weight")
    ax.plot(gens, replan, "m-", label="replan")
    ax.plot(gens, candidates, "c-", label="candidates")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Parameter Value")
    ax.set_title("Parameter Evolution (Best Individual)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = run_dir / "evolution_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Liquid War 5 AI evolution results"
    )
    parser.add_argument(
        "run_dir",
        help="Path to evolution run directory (e.g., results/20260411_200000)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate matplotlib plots"
    )

    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    print_summary(run_dir)

    if args.plot:
        plot_evolution(run_dir)


if __name__ == "__main__":
    main()
