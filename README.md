# Liquid War 5 AI Training

A GPU-native Liquid War: one PyTorch engine (`simulator/engine.py`) is **both
the playable game and the RL training environment** — you play the exact engine
the policy trains in, no fidelity gap.

## Current state (June 2026)

- **Play it**: `scripts/run-play.sh` → http://192.168.1.133:8099 — 60fps at
  384×576 with 8000 fighters/team (the whole engine tick is CUDA-graph
  captured; it is kernel-launch bound at batch=1). 9 stances × re-tap modes
  (25 combinations), cross-team Doom/Maelstrom physics, WebGL2 mote renderer.
- **LAN multiplayer**: open the same `/?room=<name>` on two machines (or use
  the 🔗 invite button) — rooms share one game, humans + AI fill the seats,
  binary delta protocol + per-client send queues keep slow WiFi guests cheap.
- **The opponent** is a PPO self-play policy (`rl/`) trained on the cluster's
  RTX PRO 6000 via ArgoCD (`charts/liquidwar-gpu-trainer`): a flat 25-action
  stance-mode head with the real cross-team wells in training (`--wells`),
  crash-proof `--resume` (a pod restart loses ≤10 updates). `best.pt` is
  selected by win-rate vs a fixed heuristic and promoted to the play server's
  checkpoint mount.
- **Big screen / PWA**: F = fullscreen, gamepad supported, installable as a
  web app over HTTPS.

The full story — engine mechanics, the capturable-tick contract, stance system,
balance history, training pipeline, and network protocol — lives in
[`docs/LIQUIDWAR_DEV.md`](docs/LIQUIDWAR_DEV.md). Cluster/machine setup:
[`docs/SETUP.md`](docs/SETUP.md).

---

## Historical: the genetic-evolution era (April 2026)

*The sections below describe the original parameter-evolution system this repo
started as (Kafka-distributed GA over the C engine's 11 scoring parameters).
That pipeline still exists but is dormant; the PPO trainer above superseded it.*

Uses a genetic algorithm to evolve 11 AI scoring parameters through self-play — pitting different parameter sets against each other across thousands of headless game simulations. Runs on a **5-node [k3s](https://k3s.io) cluster** — 104 CPU cores, 6 GPUs — distributed via Kafka.

## Architecture

```
                    ┌─────────────────────┐
                    │   Coordinator       │
                    │   coordinator.py    │
                    │   - GA selection    │
                    │   - Breeding        │
                    └──────┬──────────────┘
                           │ publishes param sets (Avro)
                           ▼
              ┌────────────────────────────┐
              │  Kafka (192.168.1.226)     │
              │  ml.liquidwar5.game-jobs   │
              └────────────────────────────┘
                           │ consumed by worker group
                           ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  5 k3s worker nodes — 104 cores, 6 GPUs (see Cluster table)    │
  │  pandoratower · pandora-storm · pandora-tank ·                 │
  │  pandoras-box · spark-wolf                                     │
  └────────────────────────────┬───────────────────────────────────┘
                           │ results (Avro)
                           ▼
              ┌────────────────────────────┐
              │  Kafka (192.168.1.226)     │
              │  ml.liquidwar5.game-results│
              └────────────┬───────────────┘
                           ▼
                    ┌─────────────────────┐
                    │   Coordinator       │
                    │   Collects, scores  │
                    │   Breeds next gen   │
                    └─────────────────────┘
```

### Cluster

Five-node [k3s](https://k3s.io) cluster, control plane on `pandoratower` (192.168.1.8). Verified via `kubectl get nodes` on 2026-05-30.

| Node | IP | Arch | Cores | RAM | GPU | k8s GPUs | k3s role |
|------|-----|------|-------|-----|-----|----------|----------|
| pandoratower | 192.168.1.8 | amd64 | 16 | 64 GB | RTX 5090 (32 GB) | 1 | control-plane |
| pandora-storm | 192.168.1.133 | amd64 | 24 | 64 GB | RTX 5090 Laptop (24 GB) | 1 | worker |
| pandora-tank | 192.168.1.222 | amd64 | 12 | 48 GB | RTX 5060 Ti ×2 (16 GB ea) | 2 | worker |
| pandoras-box | 192.168.1.226 | amd64 | 32 | 192 GB | RTX PRO 6000 (96 GB) | 0 ¹ | worker · Kafka/Schema Registry host |
| spark-wolf | 192.168.1.229 | arm64 | 20 | 128 GB | GB10 Grace-Blackwell (128 GB unified) | 1 | worker (this box) |

**Totals:** 104 CPU cores · ~496 GB RAM · 6 physical GPUs (5 exposed to k8s).

¹ pandoras-box's RTX PRO 6000 is installed but not exposed to Kubernetes (`nvidia.com/gpu` allocatable = 0) — not currently schedulable for GPU pods.

GPU VRAM figures are nominal (model spec), not measured. CUDA verified only on **spark-wolf**: driver 580.142 / runtime 13.0, toolkits 12.8 + 13.0, PyTorch 2.11.0+cu130 (`torch.cuda.is_available()` → `NVIDIA GB10`). Note `nvcc` is not on `PATH` there (it lives at `/usr/local/cuda/bin`). CUDA on the amd64 nodes has not been re-verified.

### Kafka Topics

| Topic | Purpose | Partitions | Retention |
|-------|---------|------------|-----------|
| `ml.liquidwar5.game-jobs` | Param sets for workers to evaluate | 6 | 1 day |
| `ml.liquidwar5.game-results` | Game outcomes from workers | 6 | 7 days |
| `ml.liquidwar5.evolution-state` | Compacted generation history | 1 | 30 days |

Topics managed via GitOps in [pandoras-box-data-acquisition](https://github.com/pbox-analytics/pandoras-box-data-acquisition), deployed by ArgoCD.

### Capacity

| Metric | Value |
|--------|-------|
| Nodes | 5 (k3s) |
| Total CPU cores | 104 |
| Total RAM | ~496 GB |
| GPUs | 6 physical (5 exposed to k8s) |
| Game duration (headless) | ~7 seconds |
| CPU throughput | ~47,000 games/hour |
| 8-hour CPU run | ~360,000 games |
| GPU-native engine (single GPU node) | ~47 games/sec, ~234k ticks/sec |

The CUDA-native engine (`gpu_evolve.py`) runs the whole population on one GPU node, independent of the Kafka/CPU path — best suited to the RTX 5090 / RTX PRO 6000 / GB10 nodes.

## AI Parameters (11 per team)

Each team can have its own parameter set for self-play evolution.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `candidates` | 10 | 3-30 | Target candidates to evaluate per decision |
| `density_radius` | 5 | 2-15 | Enemy density search radius (grid cells) |
| `density_weight` | 50 | 5-500 | Weight for enemy concentration in scoring |
| `health_weight` | 100 | 10-500 | Divisor for health factor in scoring |
| `replan` | 50 | 5-200 | Ticks between forced path replanning |
| `retreat` | 20 | 5-100 | Retreat threshold (retreat if lost > 1/N fighters) |
| `distance_weight` | 10 | 1-100 | Penalty for far targets (scaled /10) |
| `target_weakest` | 0 | 0-100 | Preference for attacking weaker teams |
| `aggression` | 50 | 0-100 | Retreat duration: 0=cautious, 100=never retreat |
| `frontline_bias` | 0 | 0-100 | Preference for targets near own fighters (frontline) |
| `cursor_momentum` | 0 | 0-100 | Preference for continuing in current direction |

Per-team parameters are passed via `-ai-params-file`:
```
# team 0: aggressive, cluster-targeting
density_weight 0 300
distance_weight 0 5
aggression 0 90
frontline_bias 0 60

# team 1: cautious, picks off weak teams
target_weakest 1 80
aggression 1 20
retreat 1 10
```

## Setup

See **[docs/SETUP.md](docs/SETUP.md)** for full setup instructions including:
- System dependencies and build steps per machine
- Kafka and Schema Registry connectivity
- Automated remote deployment (`deploy.sh`)
- Troubleshooting guide

Quick setup with uv:
```bash
cd liquidwar5-ai-training
uv sync
```

## Usage

### Single machine (no Kafka)

Quick test:
```bash
uv run python3 evolve.py \
    --game-binary ../liquidwar5-ai/src/liquidwar \
    --dat-path ../liquidwar5-ai/data/liquidwar.dat \
    --generations 5 --population 10 --games-per-eval 5
```

### Distributed (Kafka) — recommended

**1. Start workers** on each machine:

```bash
uv run python3 worker.py \
    --game-binary ../liquidwar5-ai/src/liquidwar \
    --dat-path ../liquidwar5-ai/data/liquidwar.dat
```

Workers auto-detect CPU count and Kafka defaults. Override with `--workers N` or `--bootstrap-servers host:port`.

**2. Start coordinator** (on any machine):

Quick run (~30 min):
```bash
uv run python3 coordinator.py \
    --generations 50 --population 30 --games-per-eval 15
```

Full 8-hour run (~360,000 games):
```bash
uv run python3 coordinator.py \
    --generations 300 --population 60 --games-per-eval 20
```

### Evolution run sizes

| Run | Pop | Games/eval | Gens | Total games | Est. time | Use case |
|-----|-----|------------|------|-------------|-----------|----------|
| Quick test | 10 | 5 | 5 | 250 | ~1 min | Verify setup works |
| Short | 20 | 10 | 30 | 6,000 | ~10 min | See initial trends |
| Solid | 30 | 15 | 50 | 22,500 | ~30 min | Good convergence |
| Full | 60 | 20 | 300 | 360,000 | ~8 hours | Deep optimization |

## Analyze Results

```bash
uv run python3 analyze.py results/<timestamp>/
uv run python3 analyze.py results/<timestamp>/ --plot  # requires matplotlib
```

## Using Evolved Parameters

Best parameters saved to `results/<timestamp>/best_params.json`:

```json
{
  "fitness": 0.82,
  "params": {
    "candidates": 18,
    "density_weight": 280,
    "distance_weight": 25,
    "target_weakest": 45,
    "aggression": 85,
    "frontline_bias": 30,
    "cursor_momentum": 15,
    "replan": 35,
    "retreat": 40
  }
}
```

Play with evolved AI:
```bash
./src/liquidwar -dat ./data/liquidwar.dat -auto \
    -ai-density-weight 280 -ai-replan 35
```

Or with per-team params file for self-play:
```bash
./src/liquidwar -dat ./data/liquidwar.dat -headless \
    -ai-params-file evolved_params.txt -teams 4
```

## Avro Schemas

All Kafka messages use Avro serialization via Confluent Schema Registry at `192.168.1.226:30081`. Schemas in `schemas/`:

| Schema | Topic | Description |
|--------|-------|-------------|
| `ai_params.avsc` | (shared) | 11-field parameter record, nested in all messages |
| `game_job.avsc` | `ml.liquidwar5.game-jobs` | Param sets for workers to evaluate |
| `game_result.avsc` | `ml.liquidwar5.game-results` | Game outcomes with per-team fighters, dominance, num_teams |
| `evolution_state.avsc` | `ml.liquidwar5.evolution-state` | Compacted generation state log |

## Files

| File | Purpose |
|------|---------|
| `evolve.py` | Standalone genetic algorithm (single machine or island model) |
| `coordinator.py` | Kafka coordinator — publishes Avro jobs, collects results, runs GA |
| `worker.py` | Kafka worker — consumes Avro jobs, runs headless games with per-team params |
| `kafka_avro.py` | Shared Avro serialization (schema loading, producer/consumer factories) |
| `analyze.py` | Results analysis and matplotlib plots |
| `schemas/` | Avro schema definitions (.avsc) |
| `docs/SETUP.md` | Full multi-machine setup guide |
| `deploy.sh` | Remote machine setup script |
| `launch_all.sh` | Launch island-model evolution on all machines |
| `machines.json` | Machine configuration template |
| `pyproject.toml` | uv project config with dependencies |

## Roadmap

- [x] Heuristic AI with scored targeting, replanning, retreat
- [x] Headless batch simulation mode
- [x] 11 configurable AI parameters per team
- [x] Per-team self-play (different params per team in same game)
- [x] Variable team count (2-6 per game)
- [x] Battle data CSV logging
- [x] Genetic algorithm parameter evolution
- [x] Distributed evolution via Kafka + Avro
- [x] 4-machine cluster deployment
- [ ] Self-play evolution (pit individuals against each other)
- [ ] Neural network opponent (replaces heuristic scoring)
- [ ] Self-play neural net training on DGX Spark
