#!/usr/bin/env python3
"""
Evolution coordinator — publishes game jobs to Kafka as Avro,
collects results, runs genetic algorithm selection, and publishes
the next generation.

Usage:
    python3 coordinator.py \
        --bootstrap-servers 192.168.1.226:31487 \
        --schema-registry http://192.168.1.226:30081 \
        --generations 50 --population 20 --games-per-eval 10
"""

import argparse
import json
import os
import random
import sys
import time

_log_file = None

def log(msg=""):
    """Log to file and stderr with immediate flush."""
    global _log_file
    line = str(msg)
    sys.stderr.write(line + "\n")
    sys.stderr.flush()
    if _log_file:
        _log_file.write(line + "\n")
        _log_file.flush()
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from kafka_avro import (
    TOPIC_JOBS, TOPIC_RESULTS, TOPIC_STATE,
    create_avro_producer, create_avro_consumer,
    produce_avro, consume_avro,
)
from evolve import AIParams


def publish_jobs(producer, serializer, key_serializer,
                 population: list, gen: int, games_per_eval: int,
                 run_id: str = "") -> int:
    """Publish game evaluation jobs for a generation."""
    job_count = 0
    now = datetime.now(timezone.utc).isoformat()

    for idx, params in enumerate(population):
        for game_num in range(games_per_eval):
            seed = gen * 100000 + idx * 1000 + game_num
            job_id = f"{run_id}_gen{gen:03d}_ind{idx:02d}_game{game_num:02d}"

            job = {
                "job_id": job_id,
                "generation": gen,
                "individual": idx,
                "game_num": game_num,
                "seed": seed,
                "params": asdict(params),
                "created_at": now,
            }

            produce_avro(producer, serializer, key_serializer,
                         TOPIC_JOBS, f"{gen}:{idx}", job)
            job_count += 1

    producer.flush()
    log(f"  Published {job_count} jobs for generation {gen}")
    return job_count


def collect_results(consumer, deserializer, key_deserializer,
                    expected_count: int,
                    timeout_seconds: int = 300,
                    run_id: str = "") -> dict:
    """Collect game results from workers.

    Returns {(gen, individual): [scores]}.
    Only accepts results whose job_id starts with our run_id.
    """
    results = {}
    received = 0
    skipped = 0
    deadline = time.time() + timeout_seconds

    while received < expected_count and time.time() < deadline:
        key, value = consume_avro(consumer, deserializer,
                                  key_deserializer, timeout=1.0)
        if value is None:
            continue

        job_id = value.get("job_id", "")
        if run_id and not job_id.startswith(run_id):
            skipped += 1
            continue

        gen = value["generation"]
        ind = value["individual"]
        score = value["result"]["dominance"]
        k = (gen, ind)

        if k not in results:
            results[k] = []
        results[k].append(score)
        received += 1

    if skipped:
        log(f"  Skipped {skipped} results from other runs")
    log(f"  Collected {received}/{expected_count} results")
    return results


def compute_fitness(results: dict, population: list,
                    gen: int) -> list:
    """Compute fitness for each individual. Returns [(params, fitness)]."""
    scored = []
    for idx, params in enumerate(population):
        scores = results.get((gen, idx), [])
        fitness = sum(scores) / len(scores) if scores else 0.0
        scored.append((params, fitness))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def publish_state(producer, serializer, key_serializer,
                  gen: int, best_params: AIParams,
                  best_fitness: float, avg_fitness: float,
                  population_size: int, games_evaluated: int,
                  duration_ms: int):
    """Publish generation state to compacted topic."""
    state = {
        "generation": gen,
        "best_fitness": best_fitness,
        "avg_fitness": avg_fitness,
        "best_params": asdict(best_params),
        "population_size": population_size,
        "games_evaluated": games_evaluated,
        "generation_duration_ms": duration_ms,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    produce_avro(producer, serializer, key_serializer,
                 TOPIC_STATE, f"gen:{gen}", state)
    producer.flush()


def run_coordinator(args):
    """Main coordinator loop."""
    jobs_prod, jobs_ser, jobs_key_ser = create_avro_producer(
        args.bootstrap_servers, args.schema_registry, TOPIC_JOBS)
    state_prod, state_ser, state_key_ser = create_avro_producer(
        args.bootstrap_servers, args.schema_registry, TOPIC_STATE)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    results_con, results_deser, results_key_deser = create_avro_consumer(
        args.bootstrap_servers, args.schema_registry, TOPIC_RESULTS,
        f"lw5-coordinator-{run_id}")

    global _log_file
    output_dir = Path(args.output) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    _log_file = open(output_dir / "coordinator.log", "w")

    log(f"=== Coordinator starting ===")
    log(f"Kafka: {args.bootstrap_servers}")
    log(f"Schema Registry: {args.schema_registry}")
    log(f"Population: {args.population}, Generations: {args.generations}")
    log(f"Games per eval: {args.games_per_eval}")
    log(f"Output: {output_dir}")
    log()

    # Initialize population
    population = [AIParams()]
    for _ in range(args.population - 1):
        population.append(AIParams.random())

    best_ever = None
    best_ever_fitness = 0.0

    for gen in range(args.generations):
        log(f"=== Generation {gen} ===")
        gen_start = time.time()

        # Publish jobs
        expected = len(population) * args.games_per_eval
        publish_jobs(jobs_prod, jobs_ser, jobs_key_ser,
                     population, gen, args.games_per_eval,
                     run_id=run_id)

        # Collect results
        results = collect_results(results_con, results_deser,
                                  results_key_deser, expected,
                                  timeout_seconds=args.timeout,
                                  run_id=run_id)

        # Compute fitness
        scored = compute_fitness(results, population, gen)

        best_params, best_fitness = scored[0]
        avg_fitness = sum(f for _, f in scored) / len(scored)
        gen_time = time.time() - gen_start
        games_total = sum(len(v) for v in results.values())

        if best_fitness > best_ever_fitness:
            best_ever_fitness = best_fitness
            best_ever = best_params

        # Publish state
        publish_state(state_prod, state_ser, state_key_ser,
                      gen, best_params, best_fitness, avg_fitness,
                      len(population), games_total,
                      int(gen_time * 1000))

        log(f"  Best: {best_fitness:.4f}  Avg: {avg_fitness:.4f}  "
              f"Time: {gen_time:.1f}s")
        log(f"  Best params: {asdict(best_params)}")
        log()

        # Save generation results locally
        gen_file = output_dir / f"generation_{gen:03d}.json"
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

    # Save best
    best_file = output_dir / "best_params.json"
    if best_ever:
        with open(best_file, "w") as f:
            json.dump({
                "fitness": best_ever_fitness,
                "params": asdict(best_ever),
                "cli_args": " ".join(best_ever.to_cli_args()),
            }, f, indent=2)

    log(f"Evolution complete!")
    log(f"Best fitness: {best_ever_fitness:.4f}")
    log(f"Best params: {asdict(best_ever)}")
    log(f"CLI: {' '.join(best_ever.to_cli_args())}")

    results_con.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evolution coordinator — publishes jobs, collects results"
    )
    parser.add_argument(
        "--bootstrap-servers", default="192.168.1.226:31487",
        help="Kafka bootstrap servers"
    )
    parser.add_argument(
        "--schema-registry", default="http://192.168.1.226:30081",
        help="Schema Registry URL"
    )
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--games-per-eval", type=int, default=10)
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Seconds to wait for results per generation"
    )
    parser.add_argument("--output", default="results")

    args = parser.parse_args()
    run_coordinator(args)


if __name__ == "__main__":
    main()
