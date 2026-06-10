#!/usr/bin/env bash
# Deploy + run the GPU Liquid War PLAY server on pandora-storm (RTX 5090
# Laptop, GPU-direct). Mounts the live simulator/ + web/ + rl/ over the
# trainer image, so a code change is just a re-run of this script.
# (Moved off pandoras-box 2026-06-09 to free the RTX PRO 6000.)
#
#   Browse:  http://192.168.1.133:8099   (or http://pandora-storm.local:8099)
#   Controls: arrows / WASD = move cursor, 1-8 = stances, T = toggle trails.
#
# Tuned play settings live in the code defaults (web/server.py):
#   grid 384x576, 8000 fighters/team, 60Hz. The one override below is
#   LW_TICK_HZ=63 — we over-target by 3Hz so asyncio's ~1ms sleep granularity
#   lands on a true 60fps (see docs/LIQUIDWAR_DEV.md, "Hitting 60fps").
#
# The policy opponent loads from /tmp/lwgood (known-good checkpoints,
# mounted over /opt/training/results). If /tmp/lwgood is empty the server
# falls back to the heuristic opponent.
set -euo pipefail

PSTORM="${PSTORM:-wolfgang@pandora-storm.local}"
IMG="${IMG:-pandoras-box.local:5000/pbox/liquidwar-gpu-trainer:0.7.1-amd64}"

cd "$(dirname "$0")/.."
rsync -rlt --no-perms --no-owner --no-group simulator web rl "$PSTORM:/tmp/lwfix/"

ssh "$PSTORM" "
  docker rm -f lwplay 2>/dev/null || true
  docker run -d --name lwplay --gpus all -p 8099:8080 \
    -v /tmp/lwfix/simulator:/opt/training/simulator \
    -v /tmp/lwfix/web:/opt/training/web \
    -v /tmp/lwfix/rl:/opt/training/rl \
    -v /tmp/lwgood:/opt/training/results \
    -e LW_PLAY_DEVICE=cuda -e LW_TICK_HZ=63 \
    --entrypoint uvicorn '$IMG' web.server:app --host 0.0.0.0 --port 8080
  sleep 6
  curl -fsS --max-time 5 http://localhost:8099/healthz && echo
"
echo 'Liquid War play server live: http://192.168.1.133:8099'
