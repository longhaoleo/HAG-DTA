#!/bin/bash
# sensitivity_n1n2_human.sh — n1/n2 网格搜索 (Human + GIN, 5 seeds, 80/10/10 单次随机划分)
#
# Usage:
#   cd ~/HAG-DTA/code_base  # Human 使用 80/10/10, seed=1234
#   bash scripts/sensitivity_n1n2_human.sh

set -e

cd "$(dirname "$0")/.."

DID=0     # Human
MID=0     # GIN

OUTPUT="${HAG_DTA_OUTPUT_ROOT:-/root/autodl-tmp/HAG-DTA-runs}"
mkdir -p "$OUTPUT/sensitivity_human"

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

run_one() {
    local n1=$1 n2=$2
    local tag="n1_${n1}_n2_${n2}"
    local log="$OUTPUT/sensitivity_human/${tag}.log"

    echo "--- Human n1=$n1 n2=$n2 ---"

    HAG_DTA_N1=$n1 HAG_DTA_N2=$n2 \
    python3 -c "
import sys
sys.argv = ['training_Human_Celegans.py', '$DID', '$MID']
exec(open('training_Human_Celegans.py').read())
" > "$log" 2>&1

    local CSV="$OUTPUT/Human_Diff_DTA_GIN_random.csv"
    if [ -f "$CSV" ]; then
        python3 -c "
import pandas as pd

df = pd.read_csv('$CSV')
for c in df.columns:
    m = df[c].mean()
    s = df[c].std(ddof=1)
    print(f'{c}={m:.4f}±{s:.4f}', end='  ')
print()
"
    else
        echo "  FAILED (no CSV)"
    fi
    echo ""
}

echo "============================================"
echo " n1/n2 Sensitivity — Human + GIN"
echo "============================================"
echo ""

for combo in "${combos[@]}"; do
    n1=$(echo $combo | awk '{print $1}')
    n2=$(echo $combo | awk '{print $2}')
    run_one "$n1" "$n2"
done

echo "============================================"
echo " All done. Logs: $OUTPUT/sensitivity_human/"
echo "============================================"
