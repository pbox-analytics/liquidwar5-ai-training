#!/usr/bin/env python3
"""
Game simulation worker — consumes jobs from Kafka, runs headless games,
publishes results back.

Run one worker per machine. Each worker uses all available CPU cores
to run games in parallel.

Usage:
    python3 worker.py \
        --bootstrap-servers pandoratower.local:30092 \
        --game-binary ../liquidwar5-ai/src/liquidwar \
        --dat-path ../liquidwar5-ai/data/liquidwar.dat \
        --workers 20
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

from confluent_kafka import Producer, Consumer, KafkaError

from evolve import AIParams

TOPIC_JOBS = "ml.liquidwar5.game-jobs"
TOPIC_RESULTS = "ml.liquidwar5.game-results"

running = True


def signal_handler(sig, frame):
    global running
    print("\nShutting down worker...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def run_game(game_binary: str, dat_path: str, params: AIParams,
             seed: int) -> float:
    """Run a single headless game. Returns fitness score."""
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
                fighters = [int(x) for x in parts[3:9]]
                total = sum(fighters)
                if total > 0:
                    return max(fighters) / total
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  Game failed (seed={seed}): {e}", file=sys.stderr)

    return 0.0


def process_job(job: dict, game_binary: str, dat_path: str) -> dict:
    """Process a single game job and return the result."""
    params = AIParams(**job["params"])
    score = run_game(game_binary, dat_path, params, job["seed"])

    return {
        "generation": job["generation"],
        "individual": job["individual"],
        "game_num": job["game_num"],
        "seed": job["seed"],
        "score": score,
        "params": job["params"],
        "worker": os.uname().nodename,
        "timestamp": time.time(),
    }


def run_worker(args):
    """Main worker loop."""
    consumer = Consumer({
        "bootstrap.servers": args.bootstrap_servers,
        "group.id": "lw5-workers",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
        "max.poll.interval.ms": 600000,  # 10 min — games take time
    })
    consumer.subscribe([TOPIC_JOBS])

    producer = Producer({
        "bootstrap.servers": args.bootstrap_servers,
        "compression.type": "snappy",
    })

    hostname = os.uname().nodename
    print(f"=== Worker starting on {hostname} ===")
    print(f"Kafka: {args.bootstrap_servers}")
    print(f"Game binary: {args.game_binary}")
    print(f"Parallel workers: {args.workers}")
    print(f"Listening on topic: {TOPIC_JOBS}")
    print()

    executor = ProcessPoolExecutor(max_workers=args.workers)
    pending_futures = []
    games_completed = 0

    while running:
        # Collect a batch of jobs
        batch = []
        for _ in range(args.batch_size):
            msg = consumer.poll(0.5)
            if msg is None:
                break
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    print(f"  Consumer error: {msg.error()}", file=sys.stderr)
                continue
            try:
                job = json.loads(msg.value().decode())
                batch.append(job)
            except json.JSONDecodeError:
                continue

        if not batch:
            # No jobs available, wait briefly
            time.sleep(0.5)
            continue

        print(f"  Processing batch of {len(batch)} jobs "
              f"(gen={batch[0].get('generation', '?')})")

        # Submit jobs to process pool
        futures = []
        for job in batch:
            future = executor.submit(
                process_job, job, args.game_binary, args.dat_path
            )
            futures.append(future)

        # Collect results and publish
        for future in futures:
            try:
                result = future.result(timeout=120)
                producer.produce(
                    TOPIC_RESULTS,
                    key=f"{result['generation']}:{result['individual']}".encode(),
                    value=json.dumps(result).encode(),
                )
                games_completed += 1
            except Exception as e:
                print(f"  Job failed: {e}", file=sys.stderr)

        producer.flush()

        if games_completed % 10 == 0:
            print(f"  [{hostname}] Total games completed: {games_completed}")

    executor.shutdown(wait=False)
    consumer.close()
    print(f"Worker shut down. Total games: {games_completed}")


def main():
    parser = argparse.ArgumentParser(
        description="Game simulation worker — run headless games from Kafka jobs"
    )
    parser.add_argument(
        "--bootstrap-servers", default="pandoratower.local:30092",
        help="Kafka bootstrap servers"
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
        "--workers", type=int, default=os.cpu_count(),
        help="Parallel game workers (default: all CPUs)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=0,
        help="Jobs to fetch per poll (default: 2x workers)"
    )

    args = parser.parse_args()
    if args.batch_size == 0:
        args.batch_size = args.workers * 2
    run_worker(args)


if __name__ == "__main__":
    main()
