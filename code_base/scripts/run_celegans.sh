#!/bin/bash
# run_celegans.sh — C.elegans + GIN | 80/10/10 单次随机划分(seed=1234) × 5 seeds
set -e

cd "$(dirname "$0")/.."
MODEL_ID=0
OUTPUT="${HAG_DTA_OUTPUT_ROOT:-/root/autodl-tmp/HAG-DTA-runs}"
mkdir -p "$OUTPUT/logs"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUTPUT/logs/celegans.log"; }

log "========== C.elegans + GIN | single split × 5 seeds =========="
set +e
python -u training_Human_Celegans.py 1 "$MODEL_ID" > "$OUTPUT/logs/celegans.log.run" 2>&1
rc=$?
set -e
cat "$OUTPUT/logs/celegans.log.run" >> "$OUTPUT/logs/celegans.log"
rm -f "$OUTPUT/logs/celegans.log.run"
[ $rc -eq 0 ] && log "DONE  C.elegans" || log "FAIL  C.elegans (exit $rc)"
exit $rc
