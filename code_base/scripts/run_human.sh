#!/bin/bash
# run_human.sh — Human + GIN | 80/10/10 单次随机划分(seed=1234) × 5 seeds
set -e

cd "$(dirname "$0")/.."
MODEL_ID=0
OUTPUT="${HAG_DTA_OUTPUT_ROOT:-/root/autodl-tmp/HAG-DTA-runs}"
mkdir -p "$OUTPUT/logs"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUTPUT/logs/human.log"; }

log "========== Human + GIN | single split × 5 seeds =========="
set +e
python -u training_Human_Celegans.py 0 "$MODEL_ID" > "$OUTPUT/logs/human.log.run" 2>&1
rc=$?
set -e
cat "$OUTPUT/logs/human.log.run" >> "$OUTPUT/logs/human.log"
rm -f "$OUTPUT/logs/human.log.run"
[ $rc -eq 0 ] && log "DONE  Human" || log "FAIL  Human (exit $rc)"
exit $rc
