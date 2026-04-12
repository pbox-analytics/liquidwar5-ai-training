#!/usr/bin/env python3
"""
Parameter evolution for Liquid War 5 AI.

Uses a genetic algorithm to evolve AI scoring parameters by running
batches of headless games and selecting parameter sets that produce
the most wins.

Supports distributed island model: multiple machines each run their
own population and periodically exchange best individuals via a shared
migrations directory.

Usage:
    # Single machine
    python3 evolve.py --game-binary ../liquidwar5-ai/src/liquidwar \
                      --dat-path ../liquidwar5-ai/data/liquidwar.dat \
                      --generations 50 --population 20 --games-per-eval 10

    # Island model (run on each machine with different island-id)
    python3 evolve.py --game-binary ../liquidwar5-ai/src/liquidwar \
                      --dat-path ../liquidwar5-ai/data/liquidwar.dat \
                      --island-id ryzen9 --migration-dir /shared/migrations

Output:
    results/<timestamp>/
        generation_NNN.csv   - raw game results per generation
        evolution_log.csv    - best/avg fitness per generation
        best_params.json     - best parameter set found
"""

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class AIParams:
    """A set of tunable AI parameters (11 per team)."""
    candidates: int = 10
    density_radius: int = 5
    density_weight: int = 50
    health_weight: int = 100
    replan: int = 50
    retreat: int = 20
    distance_weight: int = 10
    target_weakest: int = 0
    aggression: int = 50
    frontline_bias: int = 0
    cursor_momentum: int = 0

    # (field_name, min, max) for mutation and random generation
    PARAM_RANGES = [
        ("candidates", 3, 30),
        ("density_radius", 2, 15),
        ("density_weight", 5, 500),
        ("health_weight", 10, 500),
        ("replan", 5, 200),
        ("retreat", 5, 100),
        ("distance_weight", 1, 100),
        ("target_weakest", 0, 100),
        ("aggression", 0, 100),
        ("frontline_bias", 0, 100),
        ("cursor_momentum", 0, 100),
    ]

    def to_cli_args(self) -> List[str]:
        return [
            "-ai-candidates", str(self.candidates),
            "-ai-density-radius", str(self.density_radius),
            "-ai-density-weight", str(self.density_weight),
            "-ai-health-weight", str(self.health_weight),
            "-ai-replan", str(self.replan),
            "-ai-retreat", str(self.retreat),
        ]

    def to_params_file_lines(self, team: int) -> List[str]:
        """Generate per-team params file lines."""
        lines = []
        for name, _, _ in self.PARAM_RANGES:
            lines.append(f"{name} {team} {getattr(self, name)}")
        return lines

    def mutate(self, rate: float = 0.3) -> "AIParams":
        """Return a mutated copy of this parameter set."""
        def maybe_mutate(val, lo, hi):
            if random.random() < rate:
                delta = int((hi - lo) * random.gauss(0, 0.2))
                return max(lo, min(hi, val + delta))
            return val

        kwargs = {}
        for name, lo, hi in self.PARAM_RANGES:
            kwargs[name] = maybe_mutate(getattr(self, name), lo, hi)
        return AIParams(**kwargs)

    @staticmethod
    def random() -> "AIParams":
        """Generate a random parameter set."""
        kwargs = {}
        for name, lo, hi in AIParams.PARAM_RANGES:
            kwargs[name] = random.randint(lo, hi)
        return AIParams(**kwargs)

    @staticmethod
    def crossover(a: "AIParams", b: "AIParams") -> "AIParams":
        """Create a child by mixing two parents."""
        kwargs = {}
        for name, _, _ in AIParams.PARAM_RANGES:
            kwargs[name] = random.choice([getattr(a, name), getattr(b, name)])
        return AIParams(**kwargs)


@dataclass
class GameResult:
    """Result of a single headless game."""
    winner: int = -1
    ticks: int = 0
    team_fighters: List[int] = field(default_factory=list)
    params: Optional[AIParams] = None


def run_game(game_binary: str, dat_path: str, params: AIParams,
             seed: int) -> Optional[GameResult]:
    """Run a single headless game and return the result."""
    cmd = [
        game_binary,
        "-dat", dat_path,
        "-headless",
        "-seed", str(seed),
    ] + params.to_cli_args()

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        for line in proc.stdout.strip().split("\n"):
            if line.startswith("result,") and not line.startswith("result,winner"):
                parts = line.split(",")
                return GameResult(
                    winner=int(parts[1]),
                    ticks=int(parts[2]),
                    team_fighters=[int(x) for x in parts[3:9]],
                    params=params,
                )
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  Game failed (seed={seed}): {e}", file=sys.stderr)
    return None


def evaluate_params(game_binary: str, dat_path: str, params: AIParams,
                    num_games: int, base_seed: int) -> float:
    """Evaluate a parameter set by running multiple games and computing fitness.

    Fitness = average fraction of total fighters held by the winning team.
    Higher means more dominant wins.
    """
    scores = []
    for i in range(num_games):
        result = run_game(game_binary, dat_path, params, base_seed + i)
        if result and result.team_fighters:
            total = sum(result.team_fighters)
            if total > 0:
                best = max(result.team_fighters)
                scores.append(best / total)

    return sum(scores) / len(scores) if scores else 0.0


def run_generation(game_binary: str, dat_path: str,
                   population: List[AIParams], num_games: int,
                   gen_num: int, max_workers: int) -> List[tuple]:
    """Evaluate all parameter sets in a generation. Returns (params, fitness) pairs."""
    results = []
    base_seed = gen_num * 10000

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, params in enumerate(population):
            seed = base_seed + idx * num_games
            future = executor.submit(
                evaluate_params, game_binary, dat_path,
                params, num_games, seed
            )
            futures[future] = (idx, params)

        for future in as_completed(futures):
            idx, params = futures[future]
            fitness = future.result()
            results.append((params, fitness))
            print(f"  Individual {idx}: fitness={fitness:.4f} "
                  f"(cand={params.candidates} dens_w={params.density_weight} "
                  f"replan={params.replan})")

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def export_migrants(migration_dir: Path, island_id: str, gen: int,
                    elite: List[AIParams], count: int):
    """Export best individuals to the shared migration directory."""
    migration_dir.mkdir(parents=True, exist_ok=True)
    export_file = migration_dir / f"{island_id}_gen{gen:03d}.json"
    migrants = [asdict(p) for p in elite[:count]]
    with open(export_file, "w") as f:
        json.dump({"island": island_id, "generation": gen,
                   "migrants": migrants}, f)
    print(f"  Exported {len(migrants)} migrants to {export_file.name}")


def import_migrants(migration_dir: Path, island_id: str) -> List[AIParams]:
    """Import migrants from other islands."""
    migrants = []
    if not migration_dir.exists():
        return migrants

    for f in sorted(migration_dir.glob("*.json")):
        if f.stem.startswith(island_id):
            continue  # Skip our own exports
        try:
            with open(f) as fh:
                data = json.load(fh)
            for m in data.get("migrants", []):
                migrants.append(AIParams(**m))
        except (json.JSONDecodeError, TypeError):
            pass

    if migrants:
        print(f"  Imported {len(migrants)} migrants from other islands")
    return migrants


def evolve(args):
    """Main evolution loop."""
    island_id = getattr(args, "island_id", None) or ""
    migration_dir = Path(args.migration_dir) if args.migration_dir else None
    migrate_interval = args.migrate_interval

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{island_id}_{timestamp}" if island_id else timestamp
    output_dir = Path(args.output) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evolution run: {run_name}")
    if island_id:
        print(f"Island: {island_id}")
        if migration_dir:
            print(f"Migration dir: {migration_dir}")
            print(f"Migration interval: every {migrate_interval} generations")
    print(f"Output: {output_dir}")
    print(f"Population: {args.population}, Generations: {args.generations}")
    print(f"Games per eval: {args.games_per_eval}, Workers: {args.workers}")
    print()

    # Initialize population
    population = [AIParams()]  # Start with default params
    for _ in range(args.population - 1):
        population.append(AIParams.random())

    evolution_log = []
    best_ever = None
    best_ever_fitness = 0.0

    for gen in range(args.generations):
        print(f"=== Generation {gen} ===")
        gen_start = time.time()

        results = run_generation(
            args.game_binary, args.dat_path,
            population, args.games_per_eval, gen, args.workers
        )

        # Log generation results
        gen_file = output_dir / f"generation_{gen:03d}.csv"
        with open(gen_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "rank", "fitness", "candidates", "density_radius",
                "density_weight", "health_weight", "replan", "retreat"
            ])
            for rank, (params, fitness) in enumerate(results):
                writer.writerow([
                    rank, f"{fitness:.4f}",
                    params.candidates, params.density_radius,
                    params.density_weight, params.health_weight,
                    params.replan, params.retreat,
                ])

        best_params, best_fitness = results[0]
        avg_fitness = sum(f for _, f in results) / len(results)
        gen_time = time.time() - gen_start

        if best_fitness > best_ever_fitness:
            best_ever_fitness = best_fitness
            best_ever = best_params

        evolution_log.append({
            "generation": gen,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "time_seconds": gen_time,
            "best_candidates": best_params.candidates,
            "best_density_weight": best_params.density_weight,
            "best_replan": best_params.replan,
        })

        print(f"  Best: {best_fitness:.4f}  Avg: {avg_fitness:.4f}  "
              f"Time: {gen_time:.1f}s")
        print(f"  Best params: {asdict(best_params)}")
        print()

        # Selection: keep top 30%
        elite_count = max(2, args.population // 3)
        elite = [params for params, _ in results[:elite_count]]

        # Island migration
        if island_id and migration_dir and gen % migrate_interval == 0:
            export_migrants(migration_dir, island_id, gen, elite, 2)
            migrants = import_migrants(migration_dir, island_id)
            if migrants:
                elite.extend(migrants[:3])

        # Build next generation
        next_pop = list(elite[:elite_count])  # Elite survive unchanged

        while len(next_pop) < args.population:
            if random.random() < 0.7:
                # Crossover + mutate
                p1 = random.choice(elite)
                p2 = random.choice(elite)
                child = AIParams.crossover(p1, p2).mutate()
            else:
                # Mutate an elite member
                child = random.choice(elite).mutate(rate=0.5)
            next_pop.append(child)

        population = next_pop

    # Write final outputs
    log_file = output_dir / "evolution_log.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=evolution_log[0].keys())
        writer.writeheader()
        writer.writerows(evolution_log)

    best_file = output_dir / "best_params.json"
    with open(best_file, "w") as f:
        json.dump({
            "fitness": best_ever_fitness,
            "params": asdict(best_ever),
            "cli_args": " ".join(best_ever.to_cli_args()),
        }, f, indent=2)

    print(f"Evolution complete!")
    print(f"Best fitness: {best_ever_fitness:.4f}")
    print(f"Best params: {asdict(best_ever)}")
    print(f"CLI: {' '.join(best_ever.to_cli_args())}")
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evolve AI parameters for Liquid War 5"
    )
    parser.add_argument(
        "--game-binary", required=True,
        help="Path to liquidwar binary"
    )
    parser.add_argument(
        "--dat-path", required=True,
        help="Path to liquidwar.dat"
    )
    parser.add_argument(
        "--generations", type=int, default=50,
        help="Number of generations (default: 50)"
    )
    parser.add_argument(
        "--population", type=int, default=20,
        help="Population size (default: 20)"
    )
    parser.add_argument(
        "--games-per-eval", type=int, default=10,
        help="Games per parameter evaluation (default: 10)"
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(),
        help="Parallel workers (default: all CPUs)"
    )
    parser.add_argument(
        "--output", default="results",
        help="Output directory (default: results)"
    )
    parser.add_argument(
        "--island-id", default="",
        help="Island identifier for distributed evolution"
    )
    parser.add_argument(
        "--migration-dir", default=None,
        help="Shared directory for island migration exchange"
    )
    parser.add_argument(
        "--migrate-interval", type=int, default=5,
        help="Generations between migrations (default: 5)"
    )

    args = parser.parse_args()
    evolve(args)


if __name__ == "__main__":
    main()
