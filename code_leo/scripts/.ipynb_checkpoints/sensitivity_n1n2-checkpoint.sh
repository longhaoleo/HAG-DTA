#!/bin/bash
# sensitivity_n1n2.sh — n1/n2 敏感性分析 (Davis + GIN, 1 fold, 1 seed)
#
# 测试 n1 ∈ {4,5,6,7,8} × n2 ∈ {2,3,4} (n2 < n1)
# 每组只跑 1 fold + 1 seed 即可，共 11 组
#
# Usage:
#   cd ~/HAG-DTA/code_leo
#   bash sensitivity_n1n2.sh

set -e

cd "$(dirname "$0")/.."

DID=0     # Davis
MID=0     # GIN
FOLD=0

# 临时覆盖种子为单个（加速）
export HAG_DTA_SEED_OVERRIDE=100

OUTPUT="${HAG_DTA_OUTPUT_ROOT:-/root/autodl-tmp/HAG-DTA-runs}"
mkdir -p "$OUTPUT/sensitivity"

echo "============================================"
echo " n1/n2 Sensitivity Analysis — Davis + GIN"
echo "============================================"
echo ""

# 临时修改 config 的 SEEDS 为单种子
# 直接通过 patch 训练脚本的默认值实现（HAG_DTA_N1/N2 已支持）

combos=(
    "4 2"
    "5 2"
    "5 3"
    "6 2"
    "6 3"  # ← current default
    "6 4"
    "7 2"
    "7 3"
    "7 4"
    "8 3"
    "8 4"
)

for combo in "${combos[@]}"; do
    n1=$(echo $combo | awk '{print $1}')
    n2=$(echo $combo | awk '{print $2}')
    tag="n1_${n1}_n2_${n2}"
    log="$OUTPUT/sensitivity/${tag}.log"

    echo "--- n1=$n1 n2=$n2 ---"

    # 用单种子跑单个 fold
    # 临时在 config 里设 SEEDS=[100]
    HAG_DTA_N1=$n1 HAG_DTA_N2=$n2 \
    python3 -c "
import config.training as ct
ct.SEEDS = [100]
import sys
sys.argv = ['training_davis_kiba.py', '$DID', '$MID', '$FOLD']
exec(open('training_davis_kiba.py').read())
" > "$log" 2>&1

    # 提取最终指标
    CSV="$OUTPUT/davis_Diff_DTA_GIN_fold${FOLD}_random.csv"
    if [ -f "$CSV" ]; then
        result=$(python3 -c "
import pandas as pd
df = pd.read_csv('$CSV')
r = df.iloc[-1]
print(f'MSE={r[\"mse\"]:.4f} CI={r[\"ci\"]:.4f} rm²={r[\"r2\"]:.4f}')
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
