#!/usr/bin/env python3
"""
Live monitor for distributed evolution runs.

Consumes from game-results and evolution-state topics to show
real-time progress: games completed, games/sec, worker activity,
generation progress, and best fitness.

Usage:
    uv run python3 monitor.py
    uv run python3 monitor.py --offset latest  # live only
"""

import argparse
import os
import sys
import time
from collections import defaultdict

from kafka_avro import (
    TOPIC_RESULTS, TOPIC_STATE,
    create_avro_consumer, consume_avro,
)


def run_monitor(args):
    results_con, results_deser, results_key_deser = create_avro_consumer(
        args.bootstrap_servers, args.schema_registry, TOPIC_RESULTS,
        f"lw5-monitor-results-{os.getpid()}")

    state_con, state_deser, state_key_deser = create_avro_consumer(
        args.bootstrap_servers, args.schema_registry, TOPIC_STATE,
        f"lw5-monitor-state-{os.getpid()}")

    # Override auto.offset.reset
    results_con.unsubscribe()
    state_con.unsubscribe()

    from confluent_kafka import TopicPartition, OFFSET_BEGINNING, OFFSET_END

    def on_assign_results(consumer, partitions):
        for p in partitions:
            p.offset = OFFSET_BEGINNING if args.offset == "earliest" else OFFSET_END
        consumer.assign(partitions)

    def on_assign_state(consumer, partitions):
        for p in partitions:
            p.offset = OFFSET_BEGINNING if args.offset == "earliest" else OFFSET_END
        consumer.assign(partitions)

    results_con.subscribe([TOPIC_RESULTS], on_assign=on_assign_results)
    state_con.subscribe([TOPIC_STATE], on_assign=on_assign_state)

    total_games = 0
    games_by_worker = defaultdict(int)
    games_by_generation = defaultdict(int)
    current_gen = -1
    best_fitness = 0.0
    best_params = {}
    start_time = time.time()
    last_print = 0
    recent_games = []

    print("=== Liquid War 5 Evolution Monitor ===")
    print(f"Kafka: {args.bootstrap_servers}")
    print(f"Schema Registry: {args.schema_registry}")
    print(f"Reading from: {args.offset}")
    print()

    try:
        while True:
            # Poll results topic
            key, value = consume_avro(results_con, results_deser,
                                      results_key_deser, timeout=0.2)
            if value is not None:
                total_games += 1
                recent_games.append(time.time())
                worker = value.get("worker", "unknown")
                gen = value.get("generation", -1)
                games_by_worker[worker] += 1
                games_by_generation[gen] += 1

            # Poll state topic
            key, value = consume_avro(state_con, state_deser,
                                      state_key_deser, timeout=0.2)
            if value is not None:
                gen = value.get("generation", -1)
                fitness = value.get("best_fitness", 0)
                if gen > current_gen:
                    current_gen = gen
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_params = value.get("best_params", {})

            # Print update every 2 seconds
            now = time.time()
            if now - last_print >= 2:
                last_print = now
                elapsed = now - start_time

                # Games per second (last 30 seconds)
                cutoff = now - 30
                recent_games = [t for t in recent_games if t > cutoff]
                gps = len(recent_games) / 30.0 if recent_games else 0

                # Estimate
                target = 1800000
                remaining = target - total_games
                eta_hours = (remaining / (gps * 3600)) if gps > 0 else 0

                print(f"\033[2J\033[H", end="")
                print("=== Liquid War 5 Evolution Monitor ===")
                print()
                print(f"  Total games:     {total_games:>10,} / {target:,}")
                print(f"  Games/sec:       {gps:>10.1f}")
                print(f"  Elapsed:         {elapsed/3600:>10.1f} hours")
                print(f"  ETA:             {eta_hours:>10.1f} hours")
                print(f"  Generation:      {current_gen:>10} / 300")
                print(f"  Best fitness:    {best_fitness:>10.4f}")
                print()

                if games_by_worker:
                    print("  Workers:")
                    for worker, count in sorted(games_by_worker.items(),
                                                key=lambda x: -x[1]):
                        pct = count / total_games * 100 if total_games else 0
                        print(f"    {worker:20s}  {count:>8,} games  ({pct:4.1f}%)")
                    print()

                if current_gen >= 0:
                    gen_games = games_by_generation.get(current_gen, 0)
                    print(f"  Gen {current_gen}: {gen_games:,} / 6,000 results")
                    print()

                if best_params:
                    print("  Best params:")
                    for k, v in sorted(best_params.items()):
                        print(f"    {k:20s}  {v}")

                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")
    finally:
        results_con.close()
        state_con.close()


def main():
    parser = argparse.ArgumentParser(
        description="Live monitor for evolution runs"
    )
    parser.add_argument(
        "--bootstrap-servers", default="192.168.1.226:31487",
    )
    parser.add_argument(
        "--schema-registry", default="http://192.168.1.226:30081",
    )
    parser.add_argument(
        "--offset", default="earliest",
        choices=["earliest", "latest"],
        help="Start from earliest (see all) or latest (live only)"
    )
    args = parser.parse_args()
    run_monitor(args)


if __name__ == "__main__":
    main()
