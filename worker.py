#!/usr/bin/env python3
"""
Game simulation worker — consumes Avro jobs from Kafka, runs headless
games, publishes Avro results back.

Run one worker per machine. Each worker uses all available CPU cores
to run games in parallel via a process pool.

Usage:
    python3 worker.py \
        --bootstrap-servers 192.168.1.226:31487 \
        --schema-registry http://192.168.1.226:30081 \
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
             seed: int, num_teams: int = 0,
             team_params: dict = None) -> dict:
    """Run a single headless game. Returns result dict.

    If team_params is provided, it's a dict of {team_index: params_dict}
    for per-team self-play. Otherwise uses params_dict for all teams.
    """
    import random as rng
    import tempfile

    if num_teams == 0:
        rng.seed(seed)
        num_teams = rng.choice([2, 3, 4, 5, 6])

    cmd = [
        game_binary,
        "-dat", dat_path,
        "-headless",
        "-seed", str(seed),
        "-teams", str(num_teams),
    ]

    params_file = None
    if team_params:
        params_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False)
        for team_idx, tp in team_params.items():
            p = AIParams(**tp)
            for line in p.to_params_file_lines(int(team_idx)):
                params_file.write(line + "\n")
        params_file.close()
        cmd += ["-ai-params-file", params_file.name]
    else:
        params = AIParams(**params_dict)
        cmd += params.to_cli_args()

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
                    "num_teams": num_teams,
                    "duration_ms": int(time.time() * 1000) - start_ms,
                }
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  Game failed (seed={seed}): {e}", file=sys.stderr)
    finally:
        if params_file:
            import os
            os.unlink(params_file.name)

    return {
        "winner": -1,
        "ticks": 0,
        "team_fighters": [0, 0, 0, 0, 0, 0],
        "total_fighters": 0,
        "dominance": 0.0,
        "num_teams": num_teams,
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
            "num_teams": sim.get("num_teams", 6),
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
    pending_futures = {}  # future -> job_id
    last_status = time.time()

    while running:
        # Continuously consume jobs and submit to pool
        # Cap in-flight jobs to worker count to avoid oversubscription
        submitted = 0
        while len(pending_futures) < args.workers:
            key, value = consume_avro(jobs_con, jobs_deser,
                                      jobs_key_deser, timeout=0.1)
            if value is None:
                break
            future = executor.submit(
                process_job, value, args.game_binary, args.dat_path
            )
            pending_futures[future] = value.get("job_id", "?")
            submitted += 1

        if submitted > 0:
            gen = value.get("generation", "?") if value else "?"
            print(f"  Submitted {submitted} jobs (gen={gen}), "
                  f"{len(pending_futures)} in flight")

        # Publish results as they complete (non-blocking check)
        done = []
        for future in pending_futures:
            if future.done():
                done.append(future)

        for future in done:
            job_id = pending_futures.pop(future)
            try:
                result = future.result(timeout=0)
                produce_avro(
                    results_prod, results_ser, results_key_ser,
                    TOPIC_RESULTS,
                    f"{result['generation']}:{result['individual']}",
                    result,
                )
                games_completed += 1
            except Exception as e:
                print(f"  Job {job_id} failed: {e}", file=sys.stderr)

        if done:
            results_prod.flush()

        # Status update every 10 seconds
        now = time.time()
        if now - last_status >= 10:
            last_status = now
            print(f"  [{hostname}] completed: {games_completed}, "
                  f"in flight: {len(pending_futures)}")

        # Brief sleep if nothing to do
        if not submitted and not done:
            time.sleep(0.2)

    executor.shutdown(wait=False)
    jobs_con.close()
    print(f"Worker shut down. Total games: {games_completed}")


def main():
    parser = argparse.ArgumentParser(
        description="Game simulation worker — consumes Avro jobs from Kafka"
    )
    parser.add_argument(
        "--bootstrap-servers", default="192.168.1.226:31487",
        help="Kafka bootstrap servers"
    )
    parser.add_argument(
        "--schema-registry", default="http://192.168.1.226:30081",
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
