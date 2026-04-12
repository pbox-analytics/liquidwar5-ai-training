# Liquid War 5 AI Training

Parameter evolution and neural network training for [liquidwar5-ai](https://github.com/pandora-wolf-meow/liquidwar5-ai).

## Prerequisites

1. Build the game with headless mode support:
   ```bash
   cd ../liquidwar5-ai
   autoconf && ./configure && gmake
   ```

2. Python 3.10+ (no external dependencies for evolution, matplotlib optional for plots)

## Quick Start

Run a small evolution to test:
```bash
python3 evolve.py \
    --game-binary ../liquidwar5-ai/src/liquidwar \
    --dat-path ../liquidwar5-ai/data/liquidwar.dat \
    --generations 5 \
    --population 10 \
    --games-per-eval 5
```

## Full Evolution Run

```bash
python3 evolve.py \
    --game-binary ../liquidwar5-ai/src/liquidwar \
    --dat-path ../liquidwar5-ai/data/liquidwar.dat \
    --generations 50 \
    --population 20 \
    --games-per-eval 10 \
    --workers 8
```

## Analyze Results

```bash
python3 analyze.py results/<timestamp>/
python3 analyze.py results/<timestamp>/ --plot  # requires matplotlib
```

## Using Evolved Parameters

The best parameters are saved to `results/<timestamp>/best_params.json` with ready-to-use CLI args:

```bash
# Play a game with evolved AI
./src/liquidwar -dat ./data/liquidwar.dat -ai-density-weight 200 -ai-replan 20
```

## CLI Parameters

| Flag | Default | Range | Description |
|------|---------|-------|-------------|
| `-ai-candidates` | 10 | 3-30 | Target candidates to evaluate |
| `-ai-density-radius` | 5 | 2-15 | Enemy density search radius |
| `-ai-density-weight` | 50 | 5-500 | Weight for enemy density in scoring |
| `-ai-health-weight` | 100 | 10-500 | Divisor for health factor in scoring |
| `-ai-replan` | 50 | 5-200 | Ticks between forced replanning |
| `-ai-retreat` | 20 | 5-100 | Retreat if lost > 1/N fighters |

## Architecture

```
evolve.py     - Genetic algorithm: mutate params -> run games -> select winners
analyze.py    - Read results, print summaries, generate plots
results/      - Output directory (one subdirectory per evolution run)
```
