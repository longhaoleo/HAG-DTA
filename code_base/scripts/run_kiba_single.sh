#!/bin/bash
# run_kiba_single.sh — KIBA classic split, 单 seed，仅 HAG-DTA GIN
#
# Usage:
#   bash scripts/run_kiba_single.sh <seed>
#   bash scripts/run_kiba_single.sh 100

set -e

if [ $# -ne 1 ]; then
    echo "Usage: bash scripts/run_kiba_single.sh <seed>"
    echo "Example: bash scripts/run_kiba_single.sh 100"
    exit 1
fi

SEED=$1
cd "$(dirname "$0")/.."

MODEL_ID=0
OUTPUT="${HAG_DTA_OUTPUT_ROOT:-/root/autodl-tmp/HAG-DTA-runs}"
mkdir -p "$OUTPUT/logs"

LOG_NAME="kiba_seed${SEED}.log"
echo "[$(date '+%H:%M:%S')] START seed=$SEED" | tee "$OUTPUT/logs/$LOG_NAME"

python3 -c "
import config.training as ct
ct.SEEDS = [$SEED]
import sys
sys.argv = ['training_davis_kiba.py', '1', '$MODEL_ID']
exec(open('training_davis_kiba.py').read())
" >> "$OUTPUT/logs/$LOG_NAME" 2>&1

RC=$?
if [ $RC -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] DONE  seed=$SEED" | tee -a "$OUTPUT/logs/$LOG_NAME"
else
    echo "[$(date '+%H:%M:%S')] FAIL  seed=$SEED (exit $RC)" | tee -a "$OUTPUT/logs/$LOG_NAME"
fi
exit $RC
