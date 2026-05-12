#!/bin/bash
# sensitivity_mmd_davis.sh - MMD beta sensitivity (Davis + HAG-DTA GIN).
#
# Usage:
#   cd ~/HAG-DTA/code_base
#   bash scripts/sensitivity_mmd_davis.sh
#
# Optional:
#   SEED=1000 bash scripts/sensitivity_mmd_davis.sh
#   SEEDS="100 1000 2000 3000 4000" bash scripts/sensitivity_mmd_davis.sh
#   BETAS="0 0.05 1.0" bash scripts/sensitivity_mmd_davis.sh
#   ALPHA=0.05 bash scripts/sensitivity_mmd_davis.sh

set -e

cd "$(dirname "$0")/.."

DID=0    # Davis
MID=0    # GIN
SEEDS=${SEEDS:-${SEED:-100}}
N1=${N1:-4}
N2=${N2:-2}
ALPHA=${ALPHA:-0.05}
BETAS=${BETAS:-"0 0.01 0.05 0.1 0.5 1.0"}

OUTPUT="${HAG_DTA_OUTPUT_ROOT:-/root/autodl-tmp/HAG-DTA-runs}"
OUT_DIR="$OUTPUT/sensitivity_mmd_davis"
SUMMARY="$OUT_DIR/mmd_beta_davis_summary.csv"
mkdir -p "$OUT_DIR"

echo "============================================"
echo " MMD beta sensitivity - Davis + HAG-DTA GIN"
echo " seeds=$SEEDS"
echo " betas=$BETAS"
echo " n1=$N1 n2=$N2 alpha=$ALPHA"
echo "============================================"
echo ""

if [ ! -f "$SUMMARY" ]; then
    echo "dataset,beta,seed,n1,n2,alpha,mse,ci,rm2,best_epoch,log" > "$SUMMARY"
fi

for seed in $SEEDS; do
    for beta in $BETAS; do
        tag="mmd_beta_${beta}_seed_${seed}"
        log="$OUT_DIR/${tag}.log"

        echo "--- Davis seed=$seed beta=$beta ---"

        if ! HAG_DTA_N1=$N1 HAG_DTA_N2=$N2 HAG_DTA_POOL_ALPHA=$ALPHA HAG_DTA_MMD_BETA=$beta \
        python3 -c "
import config.training as ct
import sys
ct.SEEDS = [$seed]
sys.argv = ['training_davis_kiba.py', '$DID', '$MID']
exec(open('training_davis_kiba.py').read())
" > "$log" 2>&1; then
            echo "  FAILED (training crashed; see $log)"
            echo ""
            continue
        fi

        result=$(python3 -c "
import re
from pathlib import Path

text = Path('$log').read_text(errors='replace')
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
            echo "davis,$beta,$seed,$N1,$N2,$ALPHA,$mse,$ci,$rm2,$epoch,$log" >> "$SUMMARY"
            printf "  MSE=%.4f CI=%.4f rm2=%.4f epoch=%s\n" "$mse" "$ci" "$rm2" "$epoch"
        fi
        echo ""
    done
done

echo "============================================"
echo " All done. Logs: $OUT_DIR/"
echo " Summary: $SUMMARY"
echo "============================================"
