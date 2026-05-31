#!/usr/bin/env python3
"""PPO self-play trainer for the Liquid War cursor policy.

Builds the batched GPU engine + a shared CursorPolicy and runs PPO self-play:
each update collects a fixed-length rollout (every team driven by the current
policy) and does clipped-PPO minibatch SGD. Checkpoints the policy weights and
optionally publishes per-update metrics to Kafka.

Run locally (CPU correctness check):
    CUDA_VISIBLE_DEVICES="" uv run python3 -m rl.train --device cpu \
        --batch-size 8 --teams 2 --updates 3 --steps 16

Run on a GPU node (real training):
    uv run python3 -m rl.train --device cuda \
        --batch-size 512 --teams 4 --updates 2000 --steps 128
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from simulator.engine import LiquidWarEngine
from rl.policy import CursorPolicy
from rl.ppo import collect_rollout, ppo_update

# Kafka metrics publishing is optional — train fine without it.
try:
    from kafka_avro import TOPIC_STATE, create_avro_producer, produce_avro
    HAS_KAFKA = True
except ImportError:
    HAS_KAFKA = False


def parse_args():
    p = argparse.ArgumentParser(description="PPO self-play for Liquid War")
    # Environment
    p.add_argument("--batch-size", type=int, default=512,
                   help="Parallel games per rollout")
    p.add_argument("--teams", type=int, default=4)
    p.add_argument("--height", type=int, default=80)
    p.add_argument("--width", type=int, default=110)
    p.add_argument("--fighters", type=int, default=500,
                   help="Fighters per team at reset")
    p.add_argument("--grad-iters", type=int, default=4)
    # PPO
    p.add_argument("--updates", type=int, default=2000,
                   help="Number of PPO updates (each = 1 rollout + SGD)")
    p.add_argument("--steps", type=int, default=128,
                   help="Ticks collected per rollout")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    # Runtime
    p.add_argument("--device", default="cuda")
    p.add_argument("--ckpt-dir", default="results/rl")
    p.add_argument("--ckpt-every", type=int, default=50,
                   help="Save a checkpoint every N updates")
    p.add_argument("--seed", type=int, default=0)
    # Kafka (optional)
    p.add_argument("--bootstrap-servers", default="")
    p.add_argument("--schema-registry", default="")
    return p.parse_args()


def setup_kafka(bootstrap, schema_registry):
    if not (HAS_KAFKA and bootstrap and schema_registry):
        return None
    try:
        prod, ser, key = create_avro_producer(
            bootstrap, schema_registry, TOPIC_STATE)
        return (prod, ser, key)
    except Exception as e:
        print(f"Kafka disabled ({e})", flush=True)
        return None


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable -> CPU", flush=True)
        device = "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}", flush=True)

    ckpt_dir = Path(args.ckpt_dir) / time.strftime("%Y%m%d_%H%M%S")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    engine = LiquidWarEngine(
        batch_size=args.batch_size, height=args.height, width=args.width,
        num_teams=args.teams, fighters_per_team=args.fighters,
        device=device, grad_iters=args.grad_iters)
    engine.reset()

    policy = CursorPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    kafka = setup_kafka(args.bootstrap_servers, args.schema_registry)

    nparams = sum(p.numel() for p in policy.parameters())
    print(f"=== PPO self-play ===", flush=True)
    print(f"device={device} batch={args.batch_size} teams={args.teams} "
          f"map={args.height}x{args.width} steps={args.steps} "
          f"updates={args.updates} policy_params={nparams}", flush=True)
    print(f"ckpt_dir={ckpt_dir}", flush=True)

    best_return = float("-inf")
    for update in range(args.updates):
        t0 = time.time()
        rollout = collect_rollout(engine, policy, args.steps, device)
        stats = ppo_update(
            policy, optimizer, rollout, num_teams=args.teams,
            epochs=args.epochs, minibatches=args.minibatches,
            clip=args.clip, vf_coef=args.vf_coef, ent_coef=args.ent_coef)
        dt = time.time() - t0
        tps = args.steps * args.batch_size / dt if dt > 0 else 0.0
        mean_ret = stats.get("mean_return", float("nan"))

        print(f"upd {update:>4}/{args.updates}  "
              f"ret={mean_ret:+.4f}  "
              f"ploss={stats.get('policy_loss', 0):+.4f}  "
              f"vloss={stats.get('value_loss', 0):.4f}  "
              f"ent={stats.get('entropy', 0):.3f}  "
              f"{tps:.0f} env-steps/s  {dt:.1f}s", flush=True)

        if kafka is not None:
            prod, ser, key = kafka
            msg = {
                "generation": update,
                "best_fitness": float(mean_ret),
                "avg_fitness": float(mean_ret),
                "best_params": {},
                "population_size": args.batch_size,
                "games_evaluated": args.steps * args.batch_size,
                "generation_duration_ms": int(dt * 1000),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            try:
                produce_avro(prod, ser, key, TOPIC_STATE, f"upd:{update}", msg)
                prod.flush()
            except Exception as e:
                print(f"kafka publish failed: {e}", flush=True)

        if mean_ret > best_return:
            best_return = mean_ret
            torch.save(policy.state_dict(), ckpt_dir / "best.pt")

        if (update + 1) % args.ckpt_every == 0:
            torch.save(policy.state_dict(), ckpt_dir / f"upd_{update:05d}.pt")

    torch.save(policy.state_dict(), ckpt_dir / "final.pt")
    with open(ckpt_dir / "summary.json", "w") as f:
        json.dump({"updates": args.updates, "best_return": best_return,
                   "params": nparams, "args": vars(args)}, f, indent=2)
    print(f"done. best_return={best_return:.4f} ckpt={ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
