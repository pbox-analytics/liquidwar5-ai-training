#!/usr/bin/env python3
"""
Live monitor for distributed evolution runs.

Watches the coordinator log and Kafka watermarks to show real-time
progress. Much simpler than consuming Avro messages.

Usage:
    uv run python3 monitor.py
    uv run python3 monitor.py --run-dir results/20260412_023556
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from confluent_kafka import Consumer, TopicPartition


TOPIC_RESULTS = "ml.liquidwar5.game-results"


def get_kafka_stats(bootstrap_servers: str) -> dict:
    """Get message counts from Kafka watermarks."""
    try:
        c = Consumer({
            "bootstrap.servers": bootstrap_servers,
            "group.id": f"lw5-monitor-wm-{os.getpid()}",
        })
        parts = c.list_topics(
            topic=TOPIC_RESULTS).topics[TOPIC_RESULTS].partitions
        total = 0
        for pid in parts:
            tp = TopicPartition(TOPIC_RESULTS, pid)
            lo, hi = c.get_watermark_offsets(tp, timeout=5)
            total += (hi - lo)
        c.close()
        return {"total_kafka_messages": total}
    except Exception:
        return {"total_kafka_messages": -1}


def find_latest_run(results_dir: str) -> Path:
    """Find the most recent run directory."""
    results = Path(results_dir)
    if not results.exists():
        return None
    runs = sorted(results.iterdir())
    return runs[-1] if runs else None


def parse_coordinator_log(log_path: Path) -> dict:
    """Parse the coordinator log for progress info."""
    info = {
        "generations_completed": 0,
        "current_gen": -1,
        "best_fitness": 0.0,
        "avg_fitness": 0.0,
        "best_params": {},
        "last_gen_time": 0.0,
        "run_games": 0,
        "skipped": 0,
    }
    if not log_path.exists():
        return info

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("=== Generation"):
                try:
                    info["current_gen"] = int(line.split()[2])
                except (IndexError, ValueError):
                    pass
            elif line.startswith("Collected"):
                try:
                    parts = line.split()
                    collected = int(parts[1].split("/")[0])
                    info["run_games"] += collected
                    info["generations_completed"] += 1
                except (IndexError, ValueError):
                    pass
            elif line.startswith("Best:"):
                try:
                    info["best_fitness"] = float(line.split()[1])
                    avg_idx = line.index("Avg:")
                    info["avg_fitness"] = float(
                        line[avg_idx:].split()[1])
                    time_idx = line.index("Time:")
                    info["last_gen_time"] = float(
                        line[time_idx:].split()[1].rstrip("s"))
                except (IndexError, ValueError):
                    pass
            elif line.startswith("Best params:"):
                try:
                    info["best_params"] = eval(
                        line[len("Best params:"):].strip())
                except Exception:
                    pass
            elif line.startswith("Skipped"):
                try:
                    info["skipped"] += int(line.split()[1])
                except (IndexError, ValueError):
                    pass

    return info


def run_monitor(args):
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run("results")
    if not run_dir:
        print("No run directory found. Start a coordinator first.")
        sys.exit(1)

    log_path = run_dir / "coordinator.log"
    print(f"Monitoring: {run_dir}")
    print(f"Log: {log_path}")
    print()

    kafka_stats = get_kafka_stats(args.bootstrap_servers)
    start_time = time.time()
    prev_games = 0

    try:
        while True:
            info = parse_coordinator_log(log_path)

            elapsed = time.time() - start_time
            games_delta = info["run_games"] - prev_games
            prev_games = info["run_games"]

            # Estimate games/sec from last generation time
            if info["last_gen_time"] > 0 and info["generations_completed"] > 0:
                games_per_gen = 6000  # population * games_per_eval
                gps = games_per_gen / info["last_gen_time"]
            else:
                gps = 0

            target = 1800000
            remaining = target - info["run_games"]
            eta_hours = (remaining / (gps * 3600)) if gps > 0 else 0

            print(f"\033[2J\033[H", end="")
            print(f"=== Liquid War 5 Evolution Monitor ===")
            print(f"    Run: {run_dir.name}")
            print()
            print(f"  This run games:  {info['run_games']:>10,} / {target:,}")
            print(f"  Games/sec:       {gps:>10.1f}")
            print(f"  Generation:      {info['generations_completed']:>10} / 300")
            print(f"  Last gen time:   {info['last_gen_time']:>10.1f}s")
            print(f"  ETA:             {eta_hours:>10.1f} hours")
            print(f"  Best fitness:    {info['best_fitness']:>10.4f}")
            print(f"  Avg fitness:     {info['avg_fitness']:>10.4f}")
            print()

            if info["best_params"]:
                print("  Best params:")
                for k, v in sorted(info["best_params"].items()):
                    print(f"    {k:20s}  {v}")
                print()

            print(f"  Kafka total messages: {kafka_stats['total_kafka_messages']:,}")

            sys.stdout.flush()
            time.sleep(5)

            # Refresh kafka stats every 30 seconds
            if int(elapsed) % 30 == 0:
                kafka_stats = get_kafka_stats(args.bootstrap_servers)

    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Live monitor for evolution runs"
    )
    parser.add_argument(
        "--bootstrap-servers", default="192.168.1.226:31487",
    )
    parser.add_argument(
        "--run-dir", default=None,
        help="Path to run directory (default: latest in results/)"
    )
    args = parser.parse_args()
    run_monitor(args)


if __name__ == "__main__":
    main()
