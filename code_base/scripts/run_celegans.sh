#!/bin/bash
# run_celegans.sh - C.elegans, single seed, 80/10/10 split(seed=1234).
# Usage: bash scripts/run_celegans.sh <seed> [models]
#   models: gin (default), gcn, gat, sage, all

set -e

usage() {
    echo "Usage: bash scripts/run_celegans.sh <seed> [models]"
    echo "models: gin (default), gcn, gat, sage, all"
}

SEED=${1:?$(usage)}
MODELS=${2:-gin}

case "$MODELS" in
    gin|gcn|gat|sage|all) ;;
    *)
        usage
        exit 2
        ;;
esac

cd "$(dirname "$0")/.."

OUTPUT="${HAG_DTA_OUTPUT_ROOT:-/root/autodl-tmp/HAG-DTA-runs}"
mkdir -p "$OUTPUT/logs"

LOG_NAME="celegans_s${SEED}.log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUTPUT/logs/$LOG_NAME"; }

selected() {
    local name="$1"
    [[ "$MODELS" == all || "$MODELS" == "$name" ]]
}

run_hag() {
    local model_id="$1"
    local label="$2"
    log "START HAG-DTA $label"
    python -u -c "
import config.training as ct
ct.SEEDS = [$SEED]
import sys
sys.argv = ['training_Human_Celegans.py', '1', '$model_id']
exec(open('training_Human_Celegans.py').read())
" >> "$OUTPUT/logs/$LOG_NAME" 2>&1
    log "DONE  HAG-DTA $label"
}

log "========== C.elegans seed=$SEED models=$MODELS =========="

selected gin && run_hag 0 GIN
selected gcn && run_hag 1 GCN
selected gat && run_hag 2 GAT
selected sage && run_hag 3 SAGE

log "========== C.elegans seed=$SEED COMPLETE =========="
