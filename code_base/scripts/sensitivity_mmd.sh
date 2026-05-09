#!/bin/bash
# sensitivity_mmd.sh — MMD loss coefficient 消融 (Davis + GIN, classic split, 1 seed)
#
# Usage:
#   cd ~/HAG-DTA/code_base
#   bash scripts/sensitivity_mmd.sh

set -e

cd "$(dirname "$0")/.."

DID=0    # Davis
MID=0    # GIN
OUTPUT="${HAG_DTA_OUTPUT_ROOT:-/root/autodl-tmp/HAG-DTA-runs}"
mkdir -p "$OUTPUT/sensitivity_mmd"

BETAS=(0 0.01 0.05 0.1 0.5 1.0)

echo "============================================"
echo " MMD Loss Coefficient Ablation — Davis + GIN"
echo "============================================"
echo ""

for beta in "${BETAS[@]}"; do
    tag="mmd_beta_${beta}"
    log="$OUTPUT/sensitivity_mmd/${tag}.log"

    echo "--- β = $beta ---"

    HAG_DTA_N1=4 HAG_DTA_N2=2 HAG_DTA_MMD_BETA=$beta \
    python3 -c "
import config.training as ct
ct.SEEDS = [100]
import sys
sys.argv = ['training_davis_kiba.py', '$DID', '$MID']
exec(open('training_davis_kiba.py').read())
" > "$log" 2>&1

    CSV="$OUTPUT/davis_Diff_DTA_GIN_random.csv"
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
echo " All done. Logs: $OUTPUT/sensitivity_mmd/"
echo "============================================"
