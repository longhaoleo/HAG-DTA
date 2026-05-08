#!/bin/bash
# run_celegans.sh — C.elegans + GIN | 5 folds × 5 seeds = 25 runs
set -e
MODEL_ID=0
OUTPUT="${HAG_DTA_OUTPUT_ROOT:-/root/autodl-tmp/HAG-DTA-runs}"
mkdir -p "$OUTPUT/logs"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUTPUT/logs/celegans.log"; }
run_one() {
    local fold=$1
    local log_name="celegans_f${fold}.log"
    log "START fold=$fold"
    python training_Human_Celegans.py 1 "$MODEL_ID" "$fold" > "$OUTPUT/logs/$log_name" 2>&1
    local rc=$?; [ $rc -eq 0 ] && log "DONE  fold=$fold" || log "FAIL  fold=$fold (exit $rc)"
    return $rc
}
log "========== C.elegans + GIN | 5-fold × 5 seeds =========="
for fold in 0 1 2 3 4; do run_one "$fold"; done
log "========== C.ELEGANS COMPLETE =========="
