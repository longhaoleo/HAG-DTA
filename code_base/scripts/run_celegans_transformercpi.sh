#!/bin/bash
# run_celegans_transformercpi.sh — C.elegans + TransformerCPI | 80/10/10 单次随机划分(seed=1234) × 5 seeds
set -e

cd "$(dirname "$0")/.."
OUTPUT="${HAG_DTA_OUTPUT_ROOT:-/root/autodl-tmp/HAG-DTA-runs}"
mkdir -p "$OUTPUT/logs"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUTPUT/logs/celegans_transformercpi.log"; }

log "========== C.elegans + TransformerCPI | single split × 5 seeds =========="
set +e
python -u training_transformercpi.py 1 > "$OUTPUT/logs/celegans_transformercpi.log.run" 2>&1
rc=$?
set -e
cat "$OUTPUT/logs/celegans_transformercpi.log.run" >> "$OUTPUT/logs/celegans_transformercpi.log"
rm -f "$OUTPUT/logs/celegans_transformercpi.log.run"
[ $rc -eq 0 ] && log "DONE  C.elegans TransformerCPI" || log "FAIL  C.elegans TransformerCPI (exit $rc)"
exit $rc
