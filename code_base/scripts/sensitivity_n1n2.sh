#!/bin/bash
# sensitivity_n1n2.sh — n1/n2 网格搜索 (Davis + GIN, classic split, 5 seeds)
#
# Usage:
#   cd ~/HAG-DTA/code_base
#   bash scripts/sensitivity_n1n2.sh

set -e

cd "$(dirname "$0")/.."

DID=0     # Davis
MID=0     # GIN
OUTPUT="${HAG_DTA_OUTPUT_ROOT:-/root/autodl-tmp/HAG-DTA-runs}"
mkdir -p "$OUTPUT/sensitivity"

echo "============================================"
echo " n1/n2 Sensitivity Analysis — Davis + GIN"
echo "============================================"
echo ""

combos=(
    "4 2"
    "4 3"
    "5 2"
    "5 3"
    "5 4"
    "6 2"
    "6 3"
    "6 4"
    "7 2"
    "7 3"
    "7 4"
    "8 2"
    "8 3"
    "8 4"
)

for combo in "${combos[@]}"; do
    n1=$(echo $combo | awk '{print $1}')
    n2=$(echo $combo | awk '{print $2}')
    tag="n1_${n1}_n2_${n2}"
    log="$OUTPUT/sensitivity/${tag}.log"

    echo "--- n1=$n1 n2=$n2 ---"

    HAG_DTA_N1=$n1 HAG_DTA_N2=$n2 \
    python3 -c "
import sys
sys.argv = ['training_davis_kiba.py', '$DID', '$MID']
exec(open('training_davis_kiba.py').read())
" > "$log" 2>&1

    CSV="$OUTPUT/davis_Diff_DTA_GIN_classic_random.csv"
    if [ -f "$CSV" ]; then
        result=$(python3 -c "
import pandas as pd

df = pd.read_csv('$CSV')
m_mse = df['mse'].mean(); s_mse = df['mse'].std(ddof=1)
m_ci  = df['ci'].mean();  s_ci  = df['ci'].std(ddof=1)
m_r2  = df['r2'].mean();  s_r2  = df['r2'].std(ddof=1)
print(f'MSE={m_mse:.4f}±{s_mse:.4f} CI={m_ci:.4f}±{s_ci:.4f} rm²={m_r2:.4f}±{s_r2:.4f}')
")
        echo "  $result"
    else
        echo "  FAILED (no CSV)"
    fi
    echo ""
done

echo "============================================"
echo " All runs complete."
echo " Logs: $OUTPUT/sensitivity/"
echo "============================================"
