# Liquid War 5 AI Training

Parameter evolution and neural network training for [liquidwar5-ai](https://github.com/pandora-wolf-meow/liquidwar5-ai).

Uses a genetic algorithm to evolve 11 AI scoring parameters through self-play — pitting different parameter sets against each other across thousands of headless game simulations. Distributed via Kafka across 4 machines (~92 CPU cores).

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
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                  ▼
  ┌────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐
  │PandoraStorm│  │ pandoratower │  │ pandoras-box │  │spark-wolf │
  │ Ultra 9    │  │ Ryzen 7      │  │ Ryzen 9      │  │ DGX GB10  │
  │ 24 cores   │  │ 16 cores     │  │ 32 cores     │  │ 20 cores  │
  └─────┬──────┘  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘
        │                │                  │                │
        ▼                ▼                  ▼                ▼
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

| Machine | Hostname | CPU | Cores | Role |
|---------|----------|-----|-------|------|
| PandoraStorm | (local) | Intel Ultra 9 275HX | 24 | Worker + Coordinator |
| pandoratower | ptow | AMD Ryzen 7 3800X | 16 | Worker |
| pandoras-box | pbox | AMD Ryzen 9 9950X3D | 32 | Worker + Kafka/Schema Registry |
| spark-wolf | swolf | NVIDIA Grace (ARM) | 20 | Worker |

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
| Total CPU cores | ~92 |
| Game duration (headless) | ~7 seconds |
| Throughput | ~47,000 games/hour |
| 8-hour run | ~360,000 games |

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
