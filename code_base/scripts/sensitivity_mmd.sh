#!/bin/bash
# sensitivity_mmd.sh — MMD loss coefficient ablation (Davis + GIN, classic split)
#
# Usage:
#   cd ~/HAG-DTA/code_base
#   bash scripts/sensitivity_mmd.sh
#
# Optional:
#   SEED=1000 bash scripts/sensitivity_mmd.sh
#   SEEDS="100 1000 2000 3000 4000" bash scripts/sensitivity_mmd.sh
#   ALPHA=0.05 bash scripts/sensitivity_mmd.sh

set -e

cd "$(dirname "$0")/.."

DID=0    # Davis
MID=0    # GIN
SEEDS=${SEEDS:-${SEED:-100}}
N1=${N1:-6}
N2=${N2:-2}
ALPHA=${ALPHA:-0.05}
OUTPUT="${HAG_DTA_OUTPUT_ROOT:-/root/autodl-tmp/HAG-DTA-runs}"
mkdir -p "$OUTPUT/sensitivity_mmd"

BETAS=(0 0.01 0.05 0.1 0.5 1.0)
SUMMARY="$OUTPUT/sensitivity_mmd/mmd_beta_summary.csv"

echo "============================================"
echo " MMD Loss Coefficient Ablation — Davis + GIN"
echo " seeds=$SEEDS"
echo " n1=$N1 n2=$N2 alpha=$ALPHA"
echo "============================================"
echo ""

echo "beta,seed,n1,n2,alpha,mse,ci,rm2,best_epoch,log" > "$SUMMARY"

for seed in $SEEDS; do
    for beta in "${BETAS[@]}"; do
        tag="mmd_beta_${beta}_seed_${seed}"
        log="$OUTPUT/sensitivity_mmd/${tag}.log"

        echo "--- seed=$seed β=$beta ---"

        HAG_DTA_N1=$N1 HAG_DTA_N2=$N2 HAG_DTA_POOL_ALPHA=$ALPHA HAG_DTA_MMD_BETA=$beta \
        python3 -c "
import config.training as ct
ct.SEEDS = [$seed]
import sys
sys.argv = ['training_davis_kiba.py', '$DID', '$MID']
exec(open('training_davis_kiba.py').read())
" > "$log" 2>&1

        result=$(python3 -c "
import re
from pathlib import Path

log = Path('$log')
text = log.read_text(errors='replace')
matches = list(re.finditer(
    r'final test \\| epoch\\s+(?P<epoch>\\d+) \\| mse=(?P<mse>[0-9.]+) ci=(?P<ci>[0-9.]+) rm2=(?P<rm2>[0-9.]+)',
    text,
))
if not matches:
    print('FAILED')
    raise SystemExit(0)
m = matches[-1]
print(f'{m.group(\"mse\")},{m.group(\"ci\")},{m.group(\"rm2\")},{m.group(\"epoch\")}')
")
        if [ "$result" = "FAILED" ]; then
            echo "  FAILED (no final test line)"
        else
            IFS=',' read -r mse ci rm2 epoch <<< "$result"
            echo "$beta,$seed,$N1,$N2,$ALPHA,$mse,$ci,$rm2,$epoch,$log" >> "$SUMMARY"
            printf "  MSE=%.4f CI=%.4f rm2=%.4f epoch=%s\n" "$mse" "$ci" "$rm2" "$epoch"
        fi
        echo ""
    done
done

echo "============================================"
echo " All done. Logs: $OUTPUT/sensitivity_mmd/"
echo " Summary: $SUMMARY"
echo "============================================"
