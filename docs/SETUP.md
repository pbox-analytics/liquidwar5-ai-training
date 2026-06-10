# Distributed Evolution Setup Guide

This guide covers setting up all machines for the liquidwar5-ai distributed parameter evolution pipeline.

## Architecture Overview

```
┌────────────────────────────────────────────────────────┐
│                   Kafka Cluster (k3s)                  │
│               192.168.1.226:31487                 │
│                                                        │
│  Schema Registry: 192.168.1.226:30081             │
│  Topics: ml.liquidwar5.game-jobs                       │
│          ml.liquidwar5.game-results                    │
│          ml.liquidwar5.evolution-state                 │
└────────────────────────────────────────────────────────┘
        ▲                                    │
        │ results (Avro)        jobs (Avro)  │
        │                                    ▼
┌───────────────────────────────────────────────────────────┐
│  5-node k3s cluster — control plane: pandoratower (.8)      │
│  pandoratower · pandora-storm · pandora-tank ·              │
│  pandoras-box (Kafka host) · spark-wolf                     │
│  104 cores · 6 GPUs — per-node specs in Machine Roles below │
└─────────────────────────────────────────────────────────────┘
```

## Machine Roles

Verified via `kubectl get nodes` on 2026-05-30. Worker counts below are the CPU-worker process counts for the Kafka path.

| Node | IP | Arch | Cores | RAM | GPU | k3s role | CPU workers |
|------|-----|------|-------|-----|-----|----------|-------------|
| pandoratower | 192.168.1.8 | amd64 | 16 | 64 GB | RTX 5090 | control-plane | 16 |
| pandora-storm | 192.168.1.133 | amd64 | 24 | 64 GB | RTX 5090 Laptop | worker | 24 |
| pandora-tank | 192.168.1.222 | amd64 | 12 | 48 GB | RTX 5060 Ti ×2 | worker | 12 |
| pandoras-box | 192.168.1.226 | amd64 | 32 | 192 GB | RTX PRO 6000 (not exposed to k8s) | worker + Kafka/Schema Registry | 32 |
| spark-wolf | 192.168.1.229 | arm64 | 20 | 128 GB | GB10 (Grace-Blackwell, unified mem) | worker | 20 |

> **Note:** the old `dgx-spark` / `ryzen7` entries with `TBD` IPs are superseded — `dgx-spark` is now **spark-wolf** (192.168.1.229). spark-wolf is **arm64**: build the game binary natively there (see the ARM note below), do not copy an x86 binary to it.

## Prerequisites (all machines)

- Linux (Ubuntu 22.04+ or similar)
- Python 3.10+
- GCC, make, autoconf, automake
- liballegro4-dev
- Network access to 192.168.1.226:31487 (Kafka) and :30081 (Schema Registry)
- SSH key access to GitHub (for cloning repos)

## Infrastructure (already running)

These are deployed on the k3s cluster and managed via ArgoCD/GitOps:

| Service | Address | Repo |
|---------|---------|------|
| Kafka broker | 192.168.1.226:31487 | pbox-analytics/pandoras-box-data-platform |
| Schema Registry | 192.168.1.226:30081 | pbox-analytics/pandoras-box-data-platform |
| Kafka topics | Auto-created / ArgoCD | pbox-analytics/pandoras-box-data-acquisition |

**Note:** The Schema Registry deployment requires `enableServiceLinks: false` in the pod spec to avoid the Confluent `PORT` env var conflict with Kubernetes service discovery.

## Setup: pandoratower (this machine)

This machine is both coordinator and worker.

### 1. Install system dependencies

```bash
sudo apt-get update
sudo apt-get install -y build-essential autoconf automake liballegro4-dev python3 python3-pip
```

### 2. Clone repos

```bash
mkdir -p ~/repo && cd ~/repo
git clone git@github.com:pandora-wolf-meow/liquidwar5-ai.git
git clone git@github.com:pandora-wolf-meow/liquidwar5-ai-training.git
```

### 3. Build the game

```bash
cd ~/repo/liquidwar5-ai
git checkout improve-opponent-ai
autoconf && ./configure && gmake
```

### 4. Verify headless mode

```bash
./src/liquidwar -dat ./data/liquidwar.dat -headless -seed 1 2>/dev/null | grep "^result,"
```

Expected output:
```
result,winner,ticks,team0_fighters,...
result,X,24000,...
```

### 5. Install Python dependencies

```bash
cd ~/repo/liquidwar5-ai-training
pip install -r requirements.txt
```

### 6. Verify Kafka + Schema Registry connectivity

```bash
# Schema Registry
curl -s http://192.168.1.226:30081/subjects
# Should return: []

# Quick Kafka test (optional)
python3 -c "
from confluent_kafka import Producer
p = Producer({'bootstrap.servers': '192.168.1.226:31487'})
p.produce('ml.liquidwar5.game-jobs', b'test')
p.flush()
print('Kafka OK')
"
```

## Setup: Remote machines (pandoras-box, dgx-spark, ryzen7)

Run these steps on each remote machine, or use the deploy script.

### Option A: Automated deploy

From pandoratower:
```bash
cd ~/repo/liquidwar5-ai-training
./deploy.sh wolfgang@192.168.1.226    # pandoras-box
./deploy.sh wolfgang@dgx-spark-ip     # dgx-spark
./deploy.sh wolfgang@ryzen7-ip        # ryzen7
```

### Option B: Manual setup

SSH into the remote machine and run:

```bash
# 1. System deps
sudo apt-get update
sudo apt-get install -y build-essential autoconf automake liballegro4-dev python3 python3-pip

# 2. Clone repos
mkdir -p ~/repo && cd ~/repo
git clone git@github.com:pandora-wolf-meow/liquidwar5-ai.git
git clone git@github.com:pandora-wolf-meow/liquidwar5-ai-training.git

# 3. Build game
cd ~/repo/liquidwar5-ai
git checkout improve-opponent-ai
autoconf && ./configure && gmake

# 4. Verify headless
./src/liquidwar -dat ./data/liquidwar.dat -headless -seed 1 2>/dev/null | grep "^result,"

# 5. Python deps
cd ~/repo/liquidwar5-ai-training
pip install -r requirements.txt

# 6. Verify Kafka connectivity
curl -s http://192.168.1.226:30081/subjects
```

### DGX Spark note

The DGX Spark uses an ARM Grace CPU. The game binary needs to be compiled natively on the DGX — do not copy the x86 binary. The build steps are the same. If `liballegro4-dev` is not available on ARM, you may need to build Allegro from source.

## Running the Pipeline

### Start workers (on each machine)

```bash
cd ~/repo/liquidwar5-ai-training

# pandoratower (coordinator machine also runs a worker)
python3 worker.py \
    --bootstrap-servers 192.168.1.226:31487 \
    --schema-registry http://192.168.1.226:30081 \
    --game-binary ../liquidwar5-ai/src/liquidwar \
    --dat-path ../liquidwar5-ai/data/liquidwar.dat \
    --workers 20

# pandoras-box
python3 worker.py \
    --bootstrap-servers 192.168.1.226:31487 \
    --schema-registry http://192.168.1.226:30081 \
    --game-binary ../liquidwar5-ai/src/liquidwar \
    --dat-path ../liquidwar5-ai/data/liquidwar.dat \
    --workers 20

# dgx-spark
python3 worker.py \
    --bootstrap-servers 192.168.1.226:31487 \
    --schema-registry http://192.168.1.226:30081 \
    --game-binary ../liquidwar5-ai/src/liquidwar \
    --dat-path ../liquidwar5-ai/data/liquidwar.dat \
    --workers 60

# ryzen7
python3 worker.py \
    --bootstrap-servers 192.168.1.226:31487 \
    --schema-registry http://192.168.1.226:30081 \
    --game-binary ../liquidwar5-ai/src/liquidwar \
    --dat-path ../liquidwar5-ai/data/liquidwar.dat \
    --workers 12
```

### Start coordinator (on pandoratower)

```bash
cd ~/repo/liquidwar5-ai-training

# Quick run (~30 min, 22,500 games)
python3 coordinator.py \
    --bootstrap-servers 192.168.1.226:31487 \
    --schema-registry http://192.168.1.226:30081 \
    --generations 50 \
    --population 30 \
    --games-per-eval 15

# Full 8-hour run (~360,000 games)
# python3 coordinator.py \
#     --bootstrap-servers 192.168.1.226:31487 \
#     --schema-registry http://192.168.1.226:30081 \
#     --generations 300 \
#     --population 60 \
#     --games-per-eval 20
```

### Running in background

Use `nohup` or `tmux`/`screen` for long-running evolution:

```bash
# With tmux
tmux new -s worker
python3 worker.py --bootstrap-servers 192.168.1.226:31487 ...
# Ctrl-B, D to detach

# With nohup
nohup python3 worker.py --bootstrap-servers 192.168.1.226:31487 ... > worker.log 2>&1 &
```

## Standalone mode (no Kafka)

For quick local testing without the distributed infrastructure:

```bash
python3 evolve.py \
    --game-binary ../liquidwar5-ai/src/liquidwar \
    --dat-path ../liquidwar5-ai/data/liquidwar.dat \
    --generations 5 --population 10 --games-per-eval 5
```

## Monitoring

### Check worker activity

Workers log to stdout:
```
=== Worker starting on pandoras-box ===
  Processing batch of 40 jobs (gen=3)
  [pandoras-box] Total games completed: 120
```

### Check evolution progress

Coordinator logs each generation:
```
=== Generation 3 ===
  Published 200 jobs for generation 3
  Collected 200/200 results
  Best: 0.7730  Avg: 0.5613  Time: 25.1s
```

### Schema Registry

```bash
# List registered schemas
curl -s http://192.168.1.226:30081/subjects

# View a schema
curl -s http://192.168.1.226:30081/subjects/ml.liquidwar5.game-jobs-value/versions/latest
```

### Kafka topics

```bash
# From pandoratower
kubectl get kafkatopics -n kafka
```

## Analyze Results

After an evolution run:

```bash
python3 analyze.py results/<timestamp>/
python3 analyze.py results/<timestamp>/ --plot  # requires matplotlib
```

## Troubleshooting

### Worker can't connect to Kafka
- Verify `pandoratower.local` resolves: `ping pandoratower.local`
- If not, add to `/etc/hosts`: `192.168.1.8 pandoratower.local pandoratower`
- Test connectivity: `nc -zv pandoratower.local 30092`

### Schema Registry connection refused
- Check pod: `kubectl get pods -n kafka -l app=schema-registry`
- Check logs: `kubectl logs -n kafka -l app=schema-registry`
- The `enableServiceLinks: false` fix is critical — without it, Kubernetes injects a `SCHEMA_REGISTRY_PORT` env var that crashes the Confluent startup script

### Game binary fails on DGX Spark
- The DGX Spark has an ARM (Grace) CPU — build natively, don't copy x86 binaries
- `liballegro4-dev` may need to be built from source on ARM

### ArgoCD not syncing topics
- Topics auto-create when first produced to (Kafka `auto.create.topics.enable: true`)
- Manual refresh: `kubectl patch application kafka-topics -n argocd --type merge -p '{"metadata":{"annotations":{"argocd.argoproj.io/refresh":"hard"}}}'`
- Or restart repo server: `kubectl rollout restart deployment argocd-repo-server -n argocd`
