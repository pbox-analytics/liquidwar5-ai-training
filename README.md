# Liquid War 5 AI Training

Parameter evolution and neural network training for [liquidwar5-ai](https://github.com/pandora-wolf-meow/liquidwar5-ai).

Uses a genetic algorithm to evolve AI scoring parameters by running thousands of headless game simulations. Supports both single-machine and distributed execution via Kafka.

## Architecture

```
                    ┌─────────────────────┐
                    │   Coordinator       │
                    │   coordinator.py    │
                    │   - GA selection    │
                    │   - Breeding        │
                    └──────┬──────────────┘
                           │ publishes param sets
                           ▼
              ┌────────────────────────────┐
              │  Kafka: ml.liquidwar5      │
              │       .game-jobs           │
              │  (pandoratower.local:30092)│
              └────────────────────────────┘
                           │ consumed by worker group
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌─────────┐ ┌─────────┐ ┌─────────┐
         │ Worker  │ │ Worker  │ │ Worker  │
         │ ultra9  │ │ ryzen9  │ │ dgx     │
         │ 20 cores│ │ 20 cores│ │ 60 cores│
         └────┬────┘ └────┬────┘ └────┬────┘
              │            │            │
              ▼            ▼            ▼
              ┌────────────────────────────┐
              │  Kafka: ml.liquidwar5      │
              │       .game-results        │
              └────────────┬───────────────┘
                           │
                           ▼
                    ┌─────────────────────┐
                    │   Coordinator       │
                    │   Collects, scores  │
                    │   Breeds next gen   │
                    └─────────────────────┘
```

### Kafka Topics

| Topic | Purpose | Partitions | Retention |
|-------|---------|------------|-----------|
| `ml.liquidwar5.game-jobs` | Param sets for workers to evaluate | 6 | 1 day |
| `ml.liquidwar5.game-results` | Game outcomes from workers | 6 | 7 days |
| `ml.liquidwar5.evolution-state` | Compacted generation history | 1 | 30 days |

Topics are managed via GitOps in [pandoras-box-data-acquisition](https://github.com/pbox-analytics/pandoras-box-data-acquisition) and deployed by ArgoCD.

## Setup

See **[docs/SETUP.md](docs/SETUP.md)** for full setup instructions for all machines including:
- System dependencies and build steps
- Kafka and Schema Registry connectivity
- Automated remote deployment
- Troubleshooting guide

## Quick Start Prerequisites

### Game binary

Build [liquidwar5-ai](https://github.com/pandora-wolf-meow/liquidwar5-ai) with headless mode:

```bash
cd ../liquidwar5-ai
sudo apt-get install -y build-essential autoconf automake liballegro4-dev
autoconf && ./configure && gmake
```

### Python dependencies

```bash
pip install -r requirements.txt  # confluent-kafka
```

For analysis plots (optional):
```bash
pip install matplotlib
```

## Usage

### Single machine (no Kafka needed)

Quick test:
```bash
python3 evolve.py \
    --game-binary ../liquidwar5-ai/src/liquidwar \
    --dat-path ../liquidwar5-ai/data/liquidwar.dat \
    --generations 5 --population 10 --games-per-eval 5
```

Full run:
```bash
python3 evolve.py \
    --game-binary ../liquidwar5-ai/src/liquidwar \
    --dat-path ../liquidwar5-ai/data/liquidwar.dat \
    --generations 50 --population 20 --games-per-eval 10
```

### Distributed (Kafka)

**1. Start workers** on each machine:

```bash
python3 worker.py \
    --bootstrap-servers pandoratower.local:30092 \
    --game-binary ../liquidwar5-ai/src/liquidwar \
    --dat-path ../liquidwar5-ai/data/liquidwar.dat
```

Workers auto-detect CPU count. Override with `--workers N`. Workers auto-scale via Kafka consumer groups — start more to share the load.

**2. Start the coordinator** (on any machine):

```bash
python3 coordinator.py \
    --bootstrap-servers pandoratower.local:30092 \
    --generations 50 --population 20 --games-per-eval 10
```

### Deploy to a remote machine

```bash
./deploy.sh user@hostname
```

This clones both repos, installs dependencies, builds the game, and verifies headless mode works.

## Analyze Results

```bash
python3 analyze.py results/<timestamp>/
python3 analyze.py results/<timestamp>/ --plot  # generates PNG plots
```

## Using Evolved Parameters

Best parameters are saved to `results/<timestamp>/best_params.json`:

```json
{
  "fitness": 0.773,
  "params": {
    "candidates": 13,
    "density_weight": 240,
    "replan": 196
  },
  "cli_args": "-ai-candidates 13 -ai-density-weight 240 -ai-replan 196 ..."
}
```

Use them in a game:
```bash
cd ../liquidwar5-ai
./src/liquidwar -dat ./data/liquidwar.dat -ai-density-weight 240 -ai-replan 196
```

Or in auto mode to watch CPU players use the evolved params:
```bash
./src/liquidwar -dat ./data/liquidwar.dat -auto -ai-density-weight 240 -ai-replan 196
```

## AI Parameters

| Flag | Default | Range | Description |
|------|---------|-------|-------------|
| `-ai-candidates` | 10 | 3-30 | Target candidates to evaluate per decision |
| `-ai-density-radius` | 5 | 2-15 | Enemy density search radius (grid cells) |
| `-ai-density-weight` | 50 | 5-500 | Weight for enemy concentration in scoring |
| `-ai-health-weight` | 100 | 10-500 | Divisor for health factor in scoring |
| `-ai-replan` | 50 | 5-200 | Ticks between forced path replanning |
| `-ai-retreat` | 20 | 5-100 | Retreat threshold (retreat if lost > 1/N fighters) |

## Avro Schemas

All Kafka messages use Avro serialization via the Confluent Schema Registry at `pandoratower.local:30081`. Schemas are in `schemas/`:

| Schema | Topic | Description |
|--------|-------|-------------|
| `game_job.avsc` | `ml.liquidwar5.game-jobs` | Param sets for workers to evaluate |
| `game_result.avsc` | `ml.liquidwar5.game-results` | Game outcomes with fighter counts and dominance score |
| `evolution_state.avsc` | `ml.liquidwar5.evolution-state` | Compacted generation state log |
| `ai_params.avsc` | (shared) | Nested record used by all three schemas |

## Files

| File | Purpose |
|------|---------|
| `evolve.py` | Standalone genetic algorithm (single machine or island model) |
| `coordinator.py` | Kafka-based coordinator — publishes Avro jobs, collects results, runs GA |
| `worker.py` | Kafka-based worker — consumes Avro jobs, runs headless games, publishes results |
| `kafka_avro.py` | Shared Avro serialization module (schema loading, producer/consumer factories) |
| `analyze.py` | Results analysis and matplotlib plots |
| `schemas/` | Avro schema definitions (.avsc) |
| `docs/SETUP.md` | Full multi-machine setup guide |
| `deploy.sh` | Remote machine setup script |
| `launch_all.sh` | Launch island-model evolution on all machines |
| `machines.json` | Machine configuration template |
| `requirements.txt` | Python dependencies (confluent-kafka[avro], fastavro) |

## Roadmap

- [x] Heuristic AI with scored targeting, replanning, retreat
- [x] Headless batch simulation mode
- [x] Configurable AI parameters via CLI
- [x] Battle data CSV logging
- [x] Genetic algorithm parameter evolution
- [x] Distributed evolution via Kafka
- [ ] Neural network opponent (replaces heuristic scoring)
- [ ] Self-play training loop on DGX Spark
