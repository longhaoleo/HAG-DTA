#!/bin/bash
# ============================================================
# run_all_gins.sh — HAG-DTA Full Experiment Script
# ============================================================
# GIN only | 5-fold CV | 5 random seeds | 4 datasets
# Total: 4 datasets × 5 folds × 5 seeds = 100 training runs
#
# Usage:
#   cd ~/HAG-DTA/code_leo
#   bash run_all_gins.sh
#
# Output: $HAG_DTA_OUTPUT_ROOT (default /root/autodl-tmp/HAG-DTA-runs)
# ============================================================

set -e

MODEL_ID=0                          # GIN
FOLDS=(0 1 2 3 4)
REGRESSION_DATASETS=("0" "1")       # 0=davis, 1=kiba
CLASSIFICATION_DATASETS=("0" "1")   # 0=Human, 1=Celegans

OUTPUT="${HAG_DTA_OUTPUT_ROOT:-/root/autodl-tmp/HAG-DTA-runs}"
mkdir -p "$OUTPUT/logs"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$OUTPUT/logs/master.log"
}

run_one() {
    local script=$1 dataset=$2 model=$3 fold=$4
    local log_name=$(basename "$script" .py)_d${dataset}_f${fold}.log
    log "START: $script dataset=$dataset model=$model fold=$fold"
    python "$script" "$dataset" "$model" "$fold" > "$OUTPUT/logs/$log_name" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        log "DONE : $script dataset=$dataset model=$model fold=$fold"
    else
        log "FAIL : $script dataset=$dataset model=$model fold=$fold (exit $rc)"
    fi
    return $rc
}

# ============================================================
# Regression: Davis + KIBA
# ============================================================
log "========== REGRESSION TASKS (Davis + KIBA) =========="

for ds in "${REGRESSION_DATASETS[@]}"; do
    for fold in "${FOLDS[@]}"; do
        log "--- $ds (Davis=0/KIBA=1) fold=$fold ---"
        run_one "training_davis_kiba.py" "$ds" "$MODEL_ID" "$fold"
    done
done

# ============================================================
# Classification: Human + C.elegans
# ============================================================
log "========== CLASSIFICATION TASKS (Human + C.elegans) =========="

for ds in "${CLASSIFICATION_DATASETS[@]}"; do
    for fold in "${FOLDS[@]}"; do
        log "--- $ds (Human=0/C.elegans=1) fold=$fold ---"
        run_one "training_Human_Celegans.py" "$ds" "$MODEL_ID" "$fold"
    done
done

log "========== ALL EXPERIMENTS COMPLETE =========="
echo ""
echo "Results in: $OUTPUT/"
echo "Logs in   : $OUTPUT/logs/"
echo ""
echo "To check progress:  tail -f $OUTPUT/logs/master.log"
echo "To count completed: grep 'DONE' $OUTPUT/logs/master.log | wc -l"
