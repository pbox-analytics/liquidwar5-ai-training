#!/bin/bash
#
# Launch evolution on all configured machines simultaneously.
#
# Usage:
#   ./launch_all.sh [generations] [population] [games_per_eval]
#
# Prerequisites:
#   - Run deploy.sh on each machine first
#   - Edit the MACHINES array below with your hostnames
#   - Ensure shared migration directory is accessible (NFS/rsync)

set -e

GENERATIONS="${1:-50}"
POPULATION="${2:-20}"
GAMES_PER_EVAL="${3:-10}"
MIGRATION_DIR="/tmp/lw5-migrations"
REPO_DIR="~/repo"

# Edit these to match your network
MACHINES=(
    "localhost|ultra9|20"
    # "wolfgang@ryzen9-host|ryzen9|20"
    # "wolfgang@dgx-host|dgx-spark|60"
    # "wolfgang@ryzen7-host|ryzen7|12"
)

echo "=== Launching distributed evolution ==="
echo "Generations: $GENERATIONS, Population: $POPULATION, Games/eval: $GAMES_PER_EVAL"
echo ""

PIDS=()

for entry in "${MACHINES[@]}"; do
    IFS='|' read -r HOST ISLAND_ID WORKERS <<< "$entry"
    echo "Starting island '$ISLAND_ID' on $HOST ($WORKERS workers)..."

    if [ "$HOST" = "localhost" ]; then
        cd ~/repo/liquidwar5-ai-training && \
        python3 evolve.py \
            --game-binary ../liquidwar5-ai/src/liquidwar \
            --dat-path ../liquidwar5-ai/data/liquidwar.dat \
            --island-id "$ISLAND_ID" \
            --migration-dir "$MIGRATION_DIR" \
            --generations "$GENERATIONS" \
            --population "$POPULATION" \
            --games-per-eval "$GAMES_PER_EVAL" \
            --workers "$WORKERS" \
            > "results/island_${ISLAND_ID}.log" 2>&1 &
        PIDS+=($!)
    else
        ssh "$HOST" "cd $REPO_DIR/liquidwar5-ai-training && \
            mkdir -p $MIGRATION_DIR && \
            python3 evolve.py \
                --game-binary ../liquidwar5-ai/src/liquidwar \
                --dat-path ../liquidwar5-ai/data/liquidwar.dat \
                --island-id '$ISLAND_ID' \
                --migration-dir '$MIGRATION_DIR' \
                --generations $GENERATIONS \
                --population $POPULATION \
                --games-per-eval $GAMES_PER_EVAL \
                --workers $WORKERS" \
            > "results/island_${ISLAND_ID}.log" 2>&1 &
        PIDS+=($!)
    fi
done

echo ""
echo "All islands launched. PIDs: ${PIDS[*]}"
echo "Logs: results/island_*.log"
echo ""
echo "To monitor: tail -f results/island_*.log"
echo "To stop all: kill ${PIDS[*]}"

wait
echo ""
echo "=== All islands complete ==="
