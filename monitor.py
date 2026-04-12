#!/usr/bin/env python3
"""
Live monitor for distributed evolution runs.

Consumes from game-results and evolution-state topics to show
real-time progress: games completed, games/sec, worker activity,
generation progress, and best fitness.

Usage:
    python3 monitor.py
    python3 monitor.py --bootstrap-servers 192.168.1.226:31487
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

from confluent_kafka import Consumer, KafkaError

TOPIC_RESULTS = "ml.liquidwar5.game-results"
TOPIC_STATE = "ml.liquidwar5.evolution-state"


def run_monitor(args):
    consumer = Consumer({
        "bootstrap.servers": args.bootstrap_servers,
        "group.id": f"lw5-monitor-{os.getpid()}",
        "auto.offset.reset": args.offset,
        "enable.auto.commit": True,
    })
    consumer.subscribe([TOPIC_RESULTS, TOPIC_STATE])

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
    print(f"Reading from: {args.offset}")
    print()

    try:
        while True:
            msg = consumer.poll(0.5)
            if msg is None:
                pass
            elif msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    print(f"Error: {msg.error()}", file=sys.stderr)
            else:
                try:
                    value = json.loads(msg.value().decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Skip Avro-encoded messages we can't decode as JSON
                    # In production, use AvroDeserializer
                    continue

                topic = msg.topic()

                if topic == TOPIC_RESULTS:
                    total_games += 1
                    recent_games.append(time.time())
                    worker = value.get("worker", "unknown")
                    gen = value.get("generation", -1)
                    games_by_worker[worker] += 1
                    games_by_generation[gen] += 1

                elif topic == TOPIC_STATE:
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

                # Clear and print
                print(f"\033[2J\033[H", end="")  # clear screen
                print("=== Liquid War 5 Evolution Monitor ===")
                print()
                print(f"  Total games:     {total_games:,}")
                print(f"  Games/sec:       {gps:.1f}")
                print(f"  Elapsed:         {elapsed/3600:.1f} hours")
                print(f"  Generation:      {current_gen}")
                print(f"  Best fitness:    {best_fitness:.4f}")
                print()

                if games_by_worker:
                    print("  Workers:")
                    for worker, count in sorted(games_by_worker.items()):
                        print(f"    {worker:20s}  {count:>8,} games")
                    print()

                if current_gen >= 0:
                    gen_games = games_by_generation.get(current_gen, 0)
                    print(f"  Current gen {current_gen}: {gen_games} games collected")
                    print()

                if best_params:
                    print("  Best params:")
                    for k, v in sorted(best_params.items()):
                        print(f"    {k:20s}  {v}")

                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")
    finally:
        consumer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Live monitor for evolution runs"
    )
    parser.add_argument(
        "--bootstrap-servers", default="192.168.1.226:31487",
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
