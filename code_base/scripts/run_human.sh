#!/bin/bash
# run_human.sh — Human + HAG-DTA GIN, 单 seed | 80/10/10 split(seed=1234)
# Usage: bash scripts/run_human.sh <seed>

set -e

SEED=${1:?Usage: bash scripts/run_human.sh <seed>}
cd "$(dirname "$0")/.."

MODEL_ID=0
OUTPUT="${HAG_DTA_OUTPUT_ROOT:-/root/autodl-tmp/HAG-DTA-runs}"
mkdir -p "$OUTPUT/logs"

LOG_NAME="human_s${SEED}.log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUTPUT/logs/$LOG_NAME"; }

log "========== Human seed=$SEED =========="
log "START HAG-DTA GIN"
python -u -c "
import config.training as ct
ct.SEEDS = [$SEED]
import sys
sys.argv = ['training_Human_Celegans.py', '0', '$MODEL_ID']
exec(open('training_Human_Celegans.py').read())
" >> "$OUTPUT/logs/$LOG_NAME" 2>&1
log "DONE  HAG-DTA GIN"
log "========== Human seed=$SEED COMPLETE =========="
