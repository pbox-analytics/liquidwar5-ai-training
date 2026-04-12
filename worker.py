#!/usr/bin/env python3
"""
Game simulation worker — consumes Avro jobs from Kafka, runs headless
games, publishes Avro results back.

Run one worker per machine. Each worker uses all available CPU cores
to run games in parallel via a process pool.

Usage:
    python3 worker.py \
        --bootstrap-servers pandoratower.local:30092 \
        --schema-registry http://pandoratower.local:30081 \
        --game-binary ../liquidwar5-ai/src/liquidwar \
        --dat-path ../liquidwar5-ai/data/liquidwar.dat
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone

from kafka_avro import (
    TOPIC_JOBS, TOPIC_RESULTS,
    create_avro_producer, create_avro_consumer,
    produce_avro, consume_avro,
)
from evolve import AIParams

running = True


def signal_handler(sig, frame):
    global running
    print("\nShutting down worker...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def run_game(game_binary: str, dat_path: str, params_dict: dict,
             seed: int) -> dict:
    """Run a single headless game. Returns result dict."""
    params = AIParams(**params_dict)
    cmd = [
        game_binary,
        "-dat", dat_path,
        "-headless",
        "-seed", str(seed),
    ] + params.to_cli_args()

    start_ms = int(time.time() * 1000)

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        for line in proc.stdout.strip().split("\n"):
            if line.startswith("result,") and not line.startswith("result,winner"):
                parts = line.split(",")
                fighters = [int(x) for x in parts[3:9]]
                total = sum(fighters)
                winner = int(parts[1])
                ticks = int(parts[2])
                dominance = max(fighters) / total if total > 0 else 0.0

                return {
                    "winner": winner,
                    "ticks": ticks,
                    "team_fighters": fighters,
                    "total_fighters": total,
                    "dominance": dominance,
                    "duration_ms": int(time.time() * 1000) - start_ms,
                }
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  Game failed (seed={seed}): {e}", file=sys.stderr)

    return {
        "winner": -1,
        "ticks": 0,
        "team_fighters": [0, 0, 0, 0, 0, 0],
        "total_fighters": 0,
        "dominance": 0.0,
        "duration_ms": int(time.time() * 1000) - start_ms,
    }


def process_job(job: dict, game_binary: str, dat_path: str) -> dict:
    """Process a game job, returning a full result message."""
    sim = run_game(game_binary, dat_path, job["params"], job["seed"])

    return {
        "job_id": job["job_id"],
        "generation": job["generation"],
        "individual": job["individual"],
        "game_num": job["game_num"],
        "seed": job["seed"],
        "params": job["params"],
        "result": {
            "winner": sim["winner"],
            "ticks": sim["ticks"],
            "team_fighters": sim["team_fighters"],
            "total_fighters": sim["total_fighters"],
            "dominance": sim["dominance"],
        },
        "worker": os.uname().nodename,
        "duration_ms": sim["duration_ms"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def run_worker(args):
    """Main worker loop."""
    jobs_con, jobs_deser, jobs_key_deser = create_avro_consumer(
        args.bootstrap_servers, args.schema_registry, TOPIC_JOBS,
        "lw5-workers")
    results_prod, results_ser, results_key_ser = create_avro_producer(
        args.bootstrap_servers, args.schema_registry, TOPIC_RESULTS)

    hostname = os.uname().nodename
    print(f"=== Worker starting on {hostname} ===")
    print(f"Kafka: {args.bootstrap_servers}")
    print(f"Schema Registry: {args.schema_registry}")
    print(f"Game binary: {args.game_binary}")
    print(f"Parallel workers: {args.workers}")
    print(f"Batch size: {args.batch_size}")
    print()

    executor = ProcessPoolExecutor(max_workers=args.workers)
    games_completed = 0

    while running:
        # Collect a batch of jobs
        batch = []
        for _ in range(args.batch_size):
            key, value = consume_avro(jobs_con, jobs_deser,
                                      jobs_key_deser, timeout=0.5)
            if value is None:
                break
            batch.append(value)

        if not batch:
            time.sleep(0.5)
            continue

        gen = batch[0].get("generation", "?")
        print(f"  Processing batch of {len(batch)} jobs (gen={gen})")

        # Submit jobs to process pool
        futures = []
        for job in batch:
            future = executor.submit(
                process_job, job, args.game_binary, args.dat_path
            )
            futures.append(future)

        # Collect results and publish as Avro
        for future in futures:
            try:
                result = future.result(timeout=120)
                produce_avro(
                    results_prod, results_ser, results_key_ser,
                    TOPIC_RESULTS,
                    f"{result['generation']}:{result['individual']}",
                    result,
                )
                games_completed += 1
            except Exception as e:
                print(f"  Job failed: {e}", file=sys.stderr)

        results_prod.flush()

        if games_completed % 10 == 0:
            print(f"  [{hostname}] Total games completed: {games_completed}")

    executor.shutdown(wait=False)
    jobs_con.close()
    print(f"Worker shut down. Total games: {games_completed}")


def main():
    parser = argparse.ArgumentParser(
        description="Game simulation worker — consumes Avro jobs from Kafka"
    )
    parser.add_argument(
        "--bootstrap-servers", default="pandoratower.local:30092",
        help="Kafka bootstrap servers"
    )
    parser.add_argument(
        "--schema-registry", default="http://pandoratower.local:30081",
        help="Schema Registry URL"
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
