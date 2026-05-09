#!/bin/bash
# run_human_transformercpi.sh — Human + TransformerCPI | 80/10/10 单次随机划分(seed=1234) × 5 seeds
set -e

cd "$(dirname "$0")/.."
OUTPUT="${HAG_DTA_OUTPUT_ROOT:-/root/autodl-tmp/HAG-DTA-runs}"
mkdir -p "$OUTPUT/logs"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUTPUT/logs/human_transformercpi.log"; }

log "========== Human + TransformerCPI | single split × 5 seeds =========="
set +e
python -u training_transformercpi.py 0 > "$OUTPUT/logs/human_transformercpi.log.run" 2>&1
rc=$?
set -e
cat "$OUTPUT/logs/human_transformercpi.log.run" >> "$OUTPUT/logs/human_transformercpi.log"
rm -f "$OUTPUT/logs/human_transformercpi.log.run"
[ $rc -eq 0 ] && log "DONE  Human TransformerCPI" || log "FAIL  Human TransformerCPI (exit $rc)"
exit $rc
