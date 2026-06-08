#!/usr/bin/env bash
# Deploy + run the GPU Liquid War PLAY server on pandoras-box (RTX PRO 6000,
# GPU-direct — NOT an MPS slice). Mounts the live simulator/ + web/ over the
# trainer image, so a code change is just a re-run of this script.
#
#   Browse:  http://192.168.1.226:8099   (or http://pandoras-box.local:8099)
#   Controls: arrows / WASD = move cursor, SPACE = Pulse, T = toggle trails.
#
# Tuned play settings now live in the code defaults (web/server.py):
#   grid 192x288, 8000 fighters/team, 60Hz. The one override below is
#   LW_TICK_HZ=63 — we over-target by 3Hz so asyncio's ~1ms sleep granularity
#   lands on a true 60fps (see docs/LIQUIDWAR_DEV.md, "Hitting 60fps").
set -euo pipefail

PBOX="${PBOX:-pandora@pandoras-box.local}"
IMG="${IMG:-pandoras-box.local:5000/pbox/liquidwar-gpu-trainer:0.7.1-amd64}"

cd "$(dirname "$0")/.."
rsync -rlt --no-perms --no-owner --no-group simulator web "$PBOX:/tmp/lwfix/"

ssh "$PBOX" "
  docker rm -f lwplay 2>/dev/null || true
  docker run -d --name lwplay --gpus all -p 8099:8080 \
    -v /tmp/lwfix/simulator:/opt/training/simulator \
    -v /tmp/lwfix/web:/opt/training/web \
    -e LW_PLAY_DEVICE=cuda -e LW_CKPT_DIR=/tmp/nockpt -e LW_TICK_HZ=63 \
    --entrypoint uvicorn '$IMG' web.server:app --host 0.0.0.0 --port 8080
  sleep 6
  curl -fsS --max-time 5 http://localhost:8099/healthz && echo
"
echo 'Liquid War play server live: http://192.168.1.226:8099'
