#!/bin/bash
# run_davis_graphdta.sh - Davis GraphDTA baseline, single seed.
# Usage: bash scripts/run_davis_graphdta.sh <seed> [models]
#   seed: 100, 1000, 2000, 3000, 4000
#   models: gin (default), gcn, gat, sage, all

set -e

usage() {
    echo "Usage: bash scripts/run_davis_graphdta.sh <seed> [models]"
    echo "models: gin (default), gcn, gat, sage, all"
}

SEED=${1:?$(usage)}
MODELS=${2:-gin}

case "$MODELS" in
    gcn|gat|gin|sage|all) ;;
    *)
        usage
        exit 2
        ;;
esac

cd "$(dirname "$0")/.."

OUTPUT="${HAG_DTA_OUTPUT_ROOT:-/root/autodl-tmp/HAG-DTA-runs}"
mkdir -p "$OUTPUT/logs"

LOG_NAME="davis_graphdta_s${SEED}.log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUTPUT/logs/$LOG_NAME"; }

selected() {
    local name="$1"
    [[ "$MODELS" == all || "$MODELS" == "$name" ]]
}

run_graphdta() {
    local model_id="$1"
    local label="$2"
    log "START GraphDTA $label"
    python -u -c "
import config.training as ct
ct.SEEDS = [$SEED]
import sys
sys.argv = ['training_graphdta.py', '0', '$model_id']
exec(open('training_graphdta.py').read())
" >> "$OUTPUT/logs/$LOG_NAME" 2>&1
    log "DONE  GraphDTA $label"
}

log "========== Davis GraphDTA seed=$SEED models=$MODELS =========="

selected gcn && run_graphdta 0 GCN
selected gat && run_graphdta 1 GAT
selected gin && run_graphdta 2 GIN
selected sage && run_graphdta 3 SAGE

log "========== Davis GraphDTA seed=$SEED COMPLETE =========="
