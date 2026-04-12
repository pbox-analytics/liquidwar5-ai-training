#!/usr/bin/env python3
"""
GPU-accelerated parameter evolution with Kafka observability.

Runs the evolution loop on a single GPU for speed, but publishes
all game results and generation state to Kafka so Spark/Jupyter
can monitor in real-time.

Each generation:
  1. Generate parameter sets for the population
  2. Run all evaluation games on GPU (thousands at once)
  3. Publish results to Kafka (Avro)
  4. Score fitness, select, crossover, mutate -> next generation

Usage:
    uv run python3 gpu_evolve.py --generations 300 --population 60 --games-per-eval 100
"""

import argparse
import json
import os
import random
import socket
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from evolve import AIParams
from simulator.gpu_native import run_games_gpu, generate_walls

# Kafka publishing (optional — runs without it if unavailable)
try:
    from kafka_avro import (
        TOPIC_RESULTS, TOPIC_STATE,
        create_avro_producer, produce_avro,
    )
    HAS_KAFKA = True
except ImportError:
    HAS_KAFKA = False


def setup_kafka(bootstrap_servers, schema_registry):
    """Set up Kafka producers for results and state topics."""
    if not HAS_KAFKA:
        return None, None, None, None, None, None

    try:
        res_prod, res_ser, res_key = create_avro_producer(
            bootstrap_servers, schema_registry, TOPIC_RESULTS)
        state_prod, state_ser, state_key = create_avro_producer(
            bootstrap_servers, schema_registry, TOPIC_STATE)
        return res_prod, res_ser, res_key, state_prod, state_ser, state_key
    except Exception as e:
        print(f"Kafka setup failed ({e}), running without Kafka")
        return None, None, None, None, None, None


def publish_game_results(res_prod, res_ser, res_key, results,
                         population, games_per_eval, gen, island_id):
    """Publish individual game results to Kafka."""
    if res_prod is None:
        return

    now = datetime.now(timezone.utc).isoformat()
    hostname = socket.gethostname()

    for idx, params in enumerate(population):
        start = idx * games_per_eval
        end = start + games_per_eval

        for g in range(games_per_eval):
            game_idx = start + g
            dom = results['dominance'][game_idx].item()
            fighters = results['fighters_per_team'][game_idx].tolist()
            total = results['total_fighters'][game_idx].item()
            ticks = results['ticks'][game_idx].item()
            winner = results['best_team'][game_idx].item()

            msg = {
                "job_id": f"{island_id}_gen{gen:03d}_ind{idx:02d}_g{g:02d}",
                "generation": gen,
                "individual": idx,
                "game_num": g,
                "seed": gen * 100000 + idx * 1000 + g,
                "params": asdict(params),
                "result": {
                    "winner": int(winner),
                    "ticks": int(ticks),
                    "team_fighters": [int(f) for f in fighters],
                    "total_fighters": int(total),
                    "dominance": float(dom),
                    "num_teams": 4,
                },
                "worker": hostname,
                "duration_ms": 0,
                "completed_at": now,
            }

            produce_avro(res_prod, res_ser, res_key,
                         TOPIC_RESULTS, f"{gen}:{idx}", msg)

    res_prod.flush()


def publish_generation_state(state_prod, state_ser, state_key,
                             gen, best_params, best_fitness, avg_fitness,
                             population_size, games_total, duration_ms):
    """Publish generation summary to Kafka."""
    if state_prod is None:
        return

    state = {
        "generation": gen,
        "best_fitness": float(best_fitness),
        "avg_fitness": float(avg_fitness),
        "best_params": asdict(best_params),
        "population_size": population_size,
        "games_evaluated": games_total,
        "generation_duration_ms": int(duration_ms),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    produce_avro(state_prod, state_ser, state_key,
                 TOPIC_STATE, f"gen:{gen}", state)
    state_prod.flush()


def evaluate_population_gpu(population, games_per_eval, device='cuda'):
    """Evaluate all individuals in one GPU batch."""
    total_games = len(population) * games_per_eval
    walls = generate_walls(total_games, device=device)

    results = run_games_gpu(
        batch_size=total_games,
        num_teams=4,
        fighters_per_team=500,
        max_ticks=5000,
        grad_iters=4,
        walls=walls,
        device=device,
    )

    scored = []
    for idx, params in enumerate(population):
        start = idx * games_per_eval
        end = start + games_per_eval
        fitness = results['dominance'][start:end].mean().item()
        scored.append((params, fitness))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored, results


def run_evolution(args):
    """Main GPU evolution loop with Kafka publishing."""
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    island_id = f"{socket.gethostname()}_{time.strftime('%Y%m%d_%H%M%S')}"

    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name()
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        print(f"GPU: {gpu_name} ({sms} SMs)")

    # Set up Kafka
    kafka_components = setup_kafka(args.bootstrap_servers,
                                    args.schema_registry)
    res_prod, res_ser, res_key = kafka_components[:3]
    state_prod, state_ser, state_key = kafka_components[3:]

    kafka_status = "connected" if res_prod else "disabled"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"gpu_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(output_dir / "evolution.log", "w")

    def log(msg=""):
        line = str(msg)
        print(line)
        sys.stdout.flush()
        log_file.write(line + "\n")
        log_file.flush()

    total_games = args.population * args.games_per_eval

    log(f"=== GPU Evolution ===")
    log(f"Island: {island_id}")
    log(f"Population: {args.population}")
    log(f"Games per eval: {args.games_per_eval}")
    log(f"Games per generation: {total_games}")
    log(f"Generations: {args.generations}")
    log(f"Total games: {total_games * args.generations:,}")
    log(f"Device: {device}")
    log(f"Kafka: {kafka_status}")
    log(f"Output: {output_dir}")
    log()

    # Initialize population
    population = [AIParams()]
    for _ in range(args.population - 1):
        population.append(AIParams.random())

    best_ever = None
    best_ever_fitness = 0.0
    total_games_run = 0

    for gen in range(args.generations):
        gen_start = time.time()

        scored, results = evaluate_population_gpu(
            population, args.games_per_eval, device)

        gen_time = time.time() - gen_start
        best_params, best_fitness = scored[0]
        avg_fitness = sum(f for _, f in scored) / len(scored)
        gps = total_games / gen_time
        total_games_run += total_games

        if best_fitness > best_ever_fitness:
            best_ever_fitness = best_fitness
            best_ever = best_params

        log(f"Gen {gen:>3d}/{args.generations}  "
            f"best={best_fitness:.4f}  avg={avg_fitness:.4f}  "
            f"time={gen_time:.1f}s  games/s={gps:.0f}  "
            f"total={total_games_run:,}")

        # Publish to Kafka
        publish_game_results(res_prod, res_ser, res_key,
                             results, population, args.games_per_eval,
                             gen, island_id)
        publish_generation_state(state_prod, state_ser, state_key,
                                 gen, best_params, best_fitness,
                                 avg_fitness, args.population,
                                 total_games, int(gen_time * 1000))

        # Save generation locally
        gen_file = output_dir / f"gen_{gen:03d}.json"
        with open(gen_file, "w") as f:
            json.dump([{"params": asdict(p), "fitness": fit}
                       for p, fit in scored], f, indent=2)

        # Selection + breeding
        elite_count = max(2, args.population // 3)
        elite = [params for params, _ in scored[:elite_count]]

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

    # Final results
    log()
    log(f"=== Evolution Complete ===")
    log(f"Best fitness: {best_ever_fitness:.4f}")
    log(f"Best params: {asdict(best_ever)}")
    log(f"Total games: {total_games_run:,}")

    best_file = output_dir / "best_params.json"
    with open(best_file, "w") as f:
        json.dump({
            "fitness": best_ever_fitness,
            "params": asdict(best_ever),
            "cli_args": " ".join(best_ever.to_cli_args()),
            "island_id": island_id,
            "total_games": total_games_run,
        }, f, indent=2)

    log_file.close()


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated parameter evolution with Kafka")
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--population", type=int, default=60)
    parser.add_argument("--games-per-eval", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results")
    parser.add_argument(
        "--bootstrap-servers", default="192.168.1.226:31487",
        help="Kafka bootstrap servers")
    parser.add_argument(
        "--schema-registry", default="http://192.168.1.226:30081",
        help="Schema Registry URL")
    args = parser.parse_args()
    run_evolution(args)


if __name__ == "__main__":
    main()
