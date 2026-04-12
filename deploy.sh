#!/bin/bash
#
# Deploy and build liquidwar5-ai on a remote machine.
#
# Usage:
#   ./deploy.sh user@hostname [repo_dir]
#
# This script:
#   1. Clones both repos on the remote machine
#   2. Installs build dependencies
#   3. Builds the game binary
#   4. Verifies headless mode works

set -e

REMOTE="$1"
REPO_DIR="${2:-~/repo}"

if [ -z "$REMOTE" ]; then
    echo "Usage: $0 user@hostname [repo_dir]"
    echo ""
    echo "Example: $0 wolfgang@192.168.1.10"
    exit 1
fi

echo "=== Deploying to $REMOTE ==="

echo ""
echo "--- Setting up directories ---"
ssh "$REMOTE" "mkdir -p $REPO_DIR"

echo ""
echo "--- Cloning liquidwar5-ai ---"
ssh "$REMOTE" "cd $REPO_DIR && \
    (test -d liquidwar5-ai || git clone git@github.com:pandora-wolf-meow/liquidwar5-ai.git) && \
    cd liquidwar5-ai && git checkout improve-opponent-ai && git pull"

echo ""
echo "--- Cloning liquidwar5-ai-training ---"
ssh "$REMOTE" "cd $REPO_DIR && \
    (test -d liquidwar5-ai-training || git clone git@github.com:pandora-wolf-meow/liquidwar5-ai-training.git) && \
    cd liquidwar5-ai-training && git pull"

echo ""
echo "--- Installing build dependencies ---"
ssh "$REMOTE" "sudo apt-get update -qq && \
    sudo apt-get install -y -qq build-essential autoconf automake liballegro4-dev python3 2>/dev/null || \
    echo 'Note: install deps manually if not using apt'"

echo ""
echo "--- Building game binary ---"
ssh "$REMOTE" "cd $REPO_DIR/liquidwar5-ai && \
    autoconf 2>/dev/null; \
    ./configure 2>&1 | tail -3 && \
    gmake 2>&1 | tail -3"

echo ""
echo "--- Verifying headless mode ---"
RESULT=$(ssh "$REMOTE" "cd $REPO_DIR/liquidwar5-ai && \
    ./src/liquidwar -dat ./data/liquidwar.dat -headless -seed 1 2>/dev/null | grep '^result,[0-9]'")

if [ -n "$RESULT" ]; then
    echo "SUCCESS: $RESULT"
else
    echo "FAILED: headless mode did not produce results"
    exit 1
fi

echo ""
echo "=== $REMOTE is ready ==="
echo ""
echo "To start evolution on this machine:"
echo "  ssh $REMOTE \"cd $REPO_DIR/liquidwar5-ai-training && \\"
echo "    python3 evolve.py \\"
echo "      --game-binary ../liquidwar5-ai/src/liquidwar \\"
echo "      --dat-path ../liquidwar5-ai/data/liquidwar.dat \\"
echo "      --island-id \$(hostname) \\"
echo "      --migration-dir /tmp/lw5-migrations \\"
echo "      --generations 50 --population 20 --games-per-eval 10\""
