#!/bin/bash
# run_kiba.sh - KIBA, single seed.
# Usage: bash scripts/run_kiba.sh <seed> [models]
#   seed: 100, 1000, 2000, 3000, 4000
#   models: gin (default), graphdta-gcn, graphdta-gat, graphdta-gin,
#           graphdta-sage, graphdta, all

set -e

usage() {
    echo "Usage: bash scripts/run_kiba.sh <seed> [models]"
    echo "models: gin (default), graphdta-gcn, graphdta-gat, graphdta-gin, graphdta-sage, graphdta, all"
}

SEED=${1:?$(usage)}
MODELS=${2:-gin}

case "$MODELS" in
    gin|graphdta-gcn|graphdta-gat|graphdta-gin|graphdta-sage|graphdta|all) ;;
    *)
        usage
        exit 2
        ;;
esac

cd "$(dirname "$0")/.."

OUTPUT="${HAG_DTA_OUTPUT_ROOT:-/root/autodl-tmp/HAG-DTA-runs}"
mkdir -p "$OUTPUT/logs"

LOG_NAME="kiba_s${SEED}.log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUTPUT/logs/$LOG_NAME"; }

selected() {
    local name="$1"
    case "$MODELS" in
        all) return 0 ;;
        graphdta)
            [[ "$name" == graphdta-* ]]
            return
            ;;
        *)
            [[ "$MODELS" == "$name" ]]
            return
            ;;
    esac
}

run_hag_gin() {
    log "START HAG-DTA GIN"
    python -u -c "
import config.training as ct
ct.SEEDS = [$SEED]
import sys
sys.argv = ['training_davis_kiba.py', '1', '0']
exec(open('training_davis_kiba.py').read())
" >> "$OUTPUT/logs/$LOG_NAME" 2>&1
    log "DONE  HAG-DTA GIN"
}

run_graphdta() {
    local model_id="$1"
    local label="$2"
    log "START GraphDTA $label"
    python -u -c "
import config.training as ct
ct.SEEDS = [$SEED]
import sys
sys.argv = ['training_graphdta.py', '1', '$model_id']
exec(open('training_graphdta.py').read())
" >> "$OUTPUT/logs/$LOG_NAME" 2>&1
    log "DONE  GraphDTA $label"
}

log "========== KIBA seed=$SEED models=$MODELS =========="

selected gin && run_hag_gin
selected graphdta-gcn && run_graphdta 0 GCN
selected graphdta-gat && run_graphdta 1 GAT
selected graphdta-gin && run_graphdta 2 GIN
selected graphdta-sage && run_graphdta 3 SAGE

log "========== KIBA seed=$SEED COMPLETE =========="
