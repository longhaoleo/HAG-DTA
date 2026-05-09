#!/bin/bash
# run_kiba_single.sh — 单 fold + 单 seed 分布式运行 KIBA
#
# Usage:
#   bash run_kiba_single.sh <fold> <seed>
#   bash run_kiba_single.sh 0 100
#   bash run_kiba_single.sh 3 1000
#
# 可用于不同 GPU / 不同终端分别启动

set -e

if [ $# -ne 2 ]; then
    echo "Usage: bash run_kiba_single.sh <fold> <seed>"
    echo "Example: bash run_kiba_single.sh 0 100"
    exit 1
fi

FOLD=$1
SEED=$2

cd "$(dirname "$0")/.."

MODEL_ID=0
OUTPUT="${HAG_DTA_OUTPUT_ROOT:-/root/autodl-tmp/HAG-DTA-runs}"
mkdir -p "$OUTPUT/logs"

LOG_NAME="kiba_f${FOLD}_seed${SEED}.log"
echo "[$(date '+%H:%M:%S')] START fold=$FOLD seed=$SEED" | tee "$OUTPUT/logs/$LOG_NAME"

python3 -c "
import config.training as ct
ct.SEEDS = [$SEED]
import sys
sys.argv = ['training_davis_kiba.py', '1', '$MODEL_ID']
exec(open('training_davis_kiba.py').read())
" >> "$OUTPUT/logs/$LOG_NAME" 2>&1

RC=$?
if [ $RC -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] DONE  fold=$FOLD seed=$SEED" | tee -a "$OUTPUT/logs/$LOG_NAME"
else
    echo "[$(date '+%H:%M:%S')] FAIL  fold=$FOLD seed=$SEED (exit $RC)" | tee -a "$OUTPUT/logs/$LOG_NAME"
fi
