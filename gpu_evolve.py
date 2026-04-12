#!/usr/bin/env python3
"""
GPU-accelerated parameter evolution for Liquid War 5 AI.

Runs the entire evolution loop on a single GPU — no Kafka, no
distributed workers needed. One GPU replaces the whole CPU cluster.

Each generation:
  1. Generate parameter sets for the population
  2. Run all evaluation games on GPU (thousands at once)
  3. Score fitness
  4. Select, crossover, mutate -> next generation

Usage:
    uv run python3 gpu_evolve.py --generations 300 --population 60 --games-per-eval 100

    # On pandoras-box (fastest):
    CUDA_HOME=/usr TORCH_CUDA_ARCH_LIST="9.0+PTX" uv run python3 gpu_evolve.py

    # On spark-wolf (native):
    CUDA_HOME=/usr/local/cuda-13.0 uv run python3 gpu_evolve.py
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from evolve import AIParams
from simulator.gpu_native import run_games_gpu, generate_walls


def evaluate_population_gpu(population, games_per_eval, num_teams_choices,
                            device='cuda'):
    """Evaluate all individuals by running games on GPU.

    For each individual, runs games_per_eval games with random team counts.
    All games for all individuals run in one GPU batch.

    Returns list of (params, fitness) sorted by fitness descending.
    """
    total_games = len(population) * games_per_eval
    walls = generate_walls(total_games, device=device)

    # Assign random team counts per game
    team_counts = torch.tensor(
        [random.choice(num_teams_choices) for _ in range(total_games)],
        device=device)

    # Run all games at once
    results = run_games_gpu(
        batch_size=total_games,
        num_teams=4,  # max teams (unused teams get 0 fighters)
        fighters_per_team=500,
        max_ticks=5000,
        grad_iters=4,
        walls=walls,
        device=device,
    )

    # Score each individual by averaging dominance across their games
    scored = []
    for idx, params in enumerate(population):
        start = idx * games_per_eval
        end = start + games_per_eval
        individual_dom = results['dominance'][start:end]
        fitness = individual_dom.mean().item()
        scored.append((params, fitness))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def run_evolution(args):
    """Main GPU evolution loop."""
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name()
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        print(f"GPU: {gpu_name} ({sms} SMs)")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"gpu_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "evolution.log"
    log_file = open(log_path, "w")

    def log(msg=""):
        line = str(msg)
        print(line)
        sys.stdout.flush()
        log_file.write(line + "\n")
        log_file.flush()

    total_games = args.population * args.games_per_eval
    log(f"=== GPU Evolution ===")
    log(f"Population: {args.population}")
    log(f"Games per eval: {args.games_per_eval}")
    log(f"Games per generation: {total_games}")
    log(f"Generations: {args.generations}")
    log(f"Total games: {total_games * args.generations:,}")
    log(f"Device: {device}")
    log(f"Output: {output_dir}")
    log()

    # Team count choices for variety
    team_choices = [2, 3, 4]

    # Initialize population
    population = [AIParams()]  # default params
    for _ in range(args.population - 1):
        population.append(AIParams.random())

    best_ever = None
    best_ever_fitness = 0.0
    evolution_data = []

    for gen in range(args.generations):
        gen_start = time.time()

        # Evaluate
        scored = evaluate_population_gpu(
            population, args.games_per_eval, team_choices, device)

        gen_time = time.time() - gen_start
        best_params, best_fitness = scored[0]
        avg_fitness = sum(f for _, f in scored) / len(scored)
        gps = total_games / gen_time

        if best_fitness > best_ever_fitness:
            best_ever_fitness = best_fitness
            best_ever = best_params

        log(f"Gen {gen:>3d}/{args.generations}  "
            f"best={best_fitness:.4f}  avg={avg_fitness:.4f}  "
            f"time={gen_time:.1f}s  games/s={gps:.0f}")

        # Save generation data
        gen_data = {
            "generation": gen,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "time_seconds": gen_time,
            "games_per_sec": gps,
            "best_params": asdict(best_params),
        }
        evolution_data.append(gen_data)

        # Save generation results
        gen_file = output_dir / f"gen_{gen:03d}.json"
        with open(gen_file, "w") as f:
            json.dump([{"params": asdict(p), "fitness": fit}
                       for p, fit in scored], f, indent=2)

        # Selection: keep top 30%
        elite_count = max(2, args.population // 3)
        elite = [params for params, _ in scored[:elite_count]]

        # Breed next generation
        next_pop = list(elite)
        while len(next_pop) < args.population:
            if random.random() < 0.7:
                p1 = random.choice(elite)
                p2 = random.choice(elite)
                child = AIParams.crossover(p1, p2).mutate()
            else:
                child = random.choice(elite).mutate(rate=0.5)
            next_pop.append(child)

        population = next_pop

    # Save final results
    log()
    log(f"=== Evolution Complete ===")
    log(f"Best fitness: {best_ever_fitness:.4f}")
    log(f"Best params: {asdict(best_ever)}")
    log(f"CLI: {' '.join(best_ever.to_cli_args())}")

    # Save best params
    best_file = output_dir / "best_params.json"
    with open(best_file, "w") as f:
        json.dump({
            "fitness": best_ever_fitness,
            "params": asdict(best_ever),
            "cli_args": " ".join(best_ever.to_cli_args()),
        }, f, indent=2)

    # Save evolution log
    evo_file = output_dir / "evolution_data.json"
    with open(evo_file, "w") as f:
        json.dump(evolution_data, f, indent=2)

    log_file.close()


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated parameter evolution")
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--population", type=int, default=60)
    parser.add_argument("--games-per-eval", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    run_evolution(args)


if __name__ == "__main__":
    main()
