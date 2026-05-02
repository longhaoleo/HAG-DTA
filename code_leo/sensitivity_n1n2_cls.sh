#!/bin/bash
# sensitivity_n1n2_cls.sh — n1/n2 敏感性分析 (分类: Human + C.elegans, GIN, 1 fold, 1 seed)
#
# 测试 n1 ∈ {4,5,6,7,8} × n2 ∈ {2,3,4} (n2 < n1)，共 11 组 × 2 数据集 = 22 组
#
# Usage:
#   cd ~/HAG-DTA/code_leo
#   bash sensitivity_n1n2_cls.sh

set -e

MID=0     # GIN
FOLD=0

OUTPUT="${HAG_DTA_OUTPUT_ROOT:-/root/autodl-tmp/HAG-DTA-runs}"
mkdir -p "$OUTPUT/sensitivity_cls"

combos=(
    "4 2"
    "5 2"
    "5 3"
    "6 2"
    "6 3"
    "6 4"
    "7 2"
    "7 3"
    "7 4"
    "8 3"
    "8 4"
)

run_one() {
    local did=$1 dsname=$2 n1=$3 n2=$4
    local tag="${dsname}_n1_${n1}_n2_${n2}"
    local log="$OUTPUT/sensitivity_cls/${tag}.log"

    echo "--- $dsname n1=$n1 n2=$n2 ---"

    HAG_DTA_N1=$n1 HAG_DTA_N2=$n2 \
    python3 -c "
import config.training as ct
ct.SEEDS = [100]
import sys
sys.argv = ['training_Human_Celegans.py', '$did', '$MID', '$FOLD']
exec(open('training_Human_Celegans.py').read())
" > "$log" 2>&1

    local CSV="$OUTPUT/${dsname}_Diff_DTA_GIN_fold${FOLD}_random.csv"
    if [ -f "$CSV" ]; then
        python3 -c "
import pandas as pd
df = pd.read_csv('$CSV')
r = df.iloc[-1]
print(f'  AUROC={r[\"AUROC\"]:.4f}  AUPRC={r[\"AUPRC\"]:.4f}  Precision={r[\"Precision\"]:.4f}  Recall={r[\"Recall\"]:.4f}')
"
    else
        echo "  FAILED (no CSV)"
    fi
    echo ""
}

echo "============================================"
echo " n1/n2 Sensitivity — Human + C.elegans + GIN"
echo "============================================"
echo ""

# ── Human ───────────────────────────────────────
echo "========== Human =========="
for combo in "${combos[@]}"; do
    n1=$(echo $combo | awk '{print $1}')
    n2=$(echo $combo | awk '{print $2}')
    run_one 0 "Human" "$n1" "$n2"
done

# ── C.elegans ───────────────────────────────────
echo "========== C.elegans =========="
for combo in "${combos[@]}"; do
    n1=$(echo $combo | awk '{print $1}')
    n2=$(echo $combo | awk '{print $2}')
    run_one 1 "Celegans" "$n1" "$n2"
done

echo "============================================"
echo " All done. Logs: $OUTPUT/sensitivity_cls/"
echo "============================================"
