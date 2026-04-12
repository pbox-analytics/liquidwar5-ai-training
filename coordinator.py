#!/usr/bin/env python3
"""
Evolution coordinator — publishes game jobs to Kafka, collects results,
runs genetic algorithm selection, and publishes next generation.

Usage:
    python3 coordinator.py \
        --bootstrap-servers pandoratower.local:30092 \
        --generations 50 --population 20 --games-per-eval 10
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

from confluent_kafka import Producer, Consumer, KafkaError

from evolve import AIParams

TOPIC_JOBS = "ml.liquidwar5.game-jobs"
TOPIC_RESULTS = "ml.liquidwar5.game-results"
TOPIC_STATE = "ml.liquidwar5.evolution-state"


def create_producer(bootstrap_servers: str) -> Producer:
    return Producer({
        "bootstrap.servers": bootstrap_servers,
        "compression.type": "snappy",
        "linger.ms": 5,
        "batch.num.messages": 100,
    })


def create_consumer(bootstrap_servers: str, group_id: str) -> Consumer:
    return Consumer({
        "bootstrap.servers": bootstrap_servers,
        "group.id": group_id,
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
    })


def publish_jobs(producer: Producer, population: list, gen: int,
                 games_per_eval: int):
    """Publish game evaluation jobs for a generation."""
    job_count = 0
    for idx, params in enumerate(population):
        for game_num in range(games_per_eval):
            seed = gen * 100000 + idx * 1000 + game_num
            job = {
                "generation": gen,
                "individual": idx,
                "game_num": game_num,
                "seed": seed,
                "params": asdict(params),
            }
            producer.produce(
                TOPIC_JOBS,
                key=f"{gen}:{idx}".encode(),
                value=json.dumps(job).encode(),
            )
            job_count += 1

    producer.flush()
    print(f"  Published {job_count} jobs for generation {gen}")
    return job_count


def collect_results(consumer: Consumer, expected_count: int,
                    timeout_seconds: int = 300) -> dict:
    """Collect game results from workers. Returns {(gen, individual): [scores]}."""
    results = {}
    received = 0
    deadline = time.time() + timeout_seconds

    while received < expected_count and time.time() < deadline:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() != KafkaError._PARTITION_EOF:
                print(f"  Consumer error: {msg.error()}", file=sys.stderr)
            continue

        try:
            result = json.loads(msg.value().decode())
            key = (result["generation"], result["individual"])
            if key not in results:
                results[key] = []
            results[key].append(result.get("score", 0.0))
            received += 1
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Bad message: {e}", file=sys.stderr)

    print(f"  Collected {received}/{expected_count} results")
    return results


def compute_fitness(results: dict, population: list,
                    gen: int) -> list:
    """Compute fitness for each individual. Returns [(params, fitness)]."""
    scored = []
    for idx, params in enumerate(population):
        key = (gen, idx)
        scores = results.get(key, [])
        fitness = sum(scores) / len(scores) if scores else 0.0
        scored.append((params, fitness))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def publish_state(producer: Producer, gen: int, best_params: AIParams,
                  best_fitness: float, avg_fitness: float):
    """Publish generation state to compacted topic."""
    state = {
        "generation": gen,
        "best_fitness": best_fitness,
        "avg_fitness": avg_fitness,
        "best_params": asdict(best_params),
        "timestamp": time.time(),
    }
    producer.produce(
        TOPIC_STATE,
        key=f"gen:{gen}".encode(),
        value=json.dumps(state).encode(),
    )
    producer.flush()


def run_coordinator(args):
    """Main coordinator loop."""
    producer = create_producer(args.bootstrap_servers)
    consumer = create_consumer(args.bootstrap_servers, "lw5-coordinator")
    consumer.subscribe([TOPIC_RESULTS])

    output_dir = Path(args.output) / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Coordinator starting ===")
    print(f"Kafka: {args.bootstrap_servers}")
    print(f"Population: {args.population}, Generations: {args.generations}")
    print(f"Games per eval: {args.games_per_eval}")
    print(f"Output: {output_dir}")
    print()

    # Initialize population
    population = [AIParams()]
    for _ in range(args.population - 1):
        population.append(AIParams.random())

    best_ever = None
    best_ever_fitness = 0.0

    for gen in range(args.generations):
        print(f"=== Generation {gen} ===")
        gen_start = time.time()

        # Publish jobs
        expected = len(population) * args.games_per_eval
        publish_jobs(producer, population, gen, args.games_per_eval)

        # Collect results
        results = collect_results(consumer, expected,
                                  timeout_seconds=args.timeout)

        # Compute fitness
        scored = compute_fitness(results, population, gen)

        best_params, best_fitness = scored[0]
        avg_fitness = sum(f for _, f in scored) / len(scored)
        gen_time = time.time() - gen_start

        if best_fitness > best_ever_fitness:
            best_ever_fitness = best_fitness
            best_ever = best_params

        # Publish state
        publish_state(producer, gen, best_params, best_fitness, avg_fitness)

        # Log
        print(f"  Best: {best_fitness:.4f}  Avg: {avg_fitness:.4f}  "
              f"Time: {gen_time:.1f}s")
        print(f"  Best params: {asdict(best_params)}")
        print()

        # Save generation results
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

    consumer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evolution coordinator — publishes jobs, collects results"
    )
    parser.add_argument(
        "--bootstrap-servers", default="pandoratower.local:30092",
        help="Kafka bootstrap servers"
    )
    parser.add_argument(
        "--generations", type=int, default=50,
    )
    parser.add_argument(
        "--population", type=int, default=20,
    )
    parser.add_argument(
        "--games-per-eval", type=int, default=10,
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Seconds to wait for results per generation"
    )
    parser.add_argument(
        "--output", default="results",
    )

    args = parser.parse_args()
    run_coordinator(args)


if __name__ == "__main__":
    main()
