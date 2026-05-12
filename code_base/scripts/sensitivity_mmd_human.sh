#!/bin/bash
# sensitivity_mmd_human.sh - MMD beta sensitivity (Human + HAG-DTA GIN).
#
# Usage:
#   cd ~/HAG-DTA/code_base
#   bash scripts/sensitivity_mmd_human.sh
#
# Optional:
#   SEED=1000 bash scripts/sensitivity_mmd_human.sh
#   SEEDS="100 1000 2000 3000 4000" bash scripts/sensitivity_mmd_human.sh
#   BETAS="0 0.05 1.0" bash scripts/sensitivity_mmd_human.sh
#   ALPHA=0.05 bash scripts/sensitivity_mmd_human.sh

set -e

cd "$(dirname "$0")/.."

DID=0    # Human
MID=0    # GIN
SEEDS=${SEEDS:-${SEED:-100}}
N1=${N1:-7}
N2=${N2:-3}
ALPHA=${ALPHA:-0.05}
BETAS=${BETAS:-"0 0.01 0.05 0.1 0.5 1.0"}

OUTPUT="${HAG_DTA_OUTPUT_ROOT:-/root/autodl-tmp/HAG-DTA-runs}"
OUT_DIR="$OUTPUT/sensitivity_mmd_human"
SUMMARY="$OUT_DIR/mmd_beta_human_summary.csv"
mkdir -p "$OUT_DIR"

echo "============================================"
echo " MMD beta sensitivity - Human + HAG-DTA GIN"
echo " seeds=$SEEDS"
echo " betas=$BETAS"
echo " n1=$N1 n2=$N2 alpha=$ALPHA"
echo "============================================"
echo ""

if [ ! -f "$SUMMARY" ]; then
    echo "dataset,beta,seed,n1,n2,alpha,AUROC,AUPRC,Precision,Recall,log" > "$SUMMARY"
fi

for seed in $SEEDS; do
    for beta in $BETAS; do
        tag="mmd_beta_${beta}_seed_${seed}"
        log="$OUT_DIR/${tag}.log"

        echo "--- Human seed=$seed beta=$beta ---"

        HAG_DTA_N1=$N1 HAG_DTA_N2=$N2 HAG_DTA_POOL_ALPHA=$ALPHA HAG_DTA_MMD_BETA=$beta \
        python3 -c "
import config.training as ct
import sys
ct.SEEDS = [$seed]
sys.argv = ['training_Human_Celegans.py', '$DID', '$MID']
exec(open('training_Human_Celegans.py').read())
" > "$log" 2>&1

        result=$(python3 -c "
import pandas as pd
from pathlib import Path

csv_path = Path('$OUTPUT') / 'Human_Diff_DTA_GIN_random.csv'
if not csv_path.exists():
    print('FAILED')
    raise SystemExit(0)
df = pd.read_csv(csv_path)
if df.empty:
    print('FAILED')
    raise SystemExit(0)
row = df.iloc[-1]
print(f'{row[\"AUROC\"]},{row[\"AUPRC\"]},{row[\"Precision\"]},{row[\"Recall\"]}')
")
        if [ "$result" = "FAILED" ]; then
            echo "  FAILED (no Human_Diff_DTA_GIN_random.csv)"
        else
            IFS=',' read -r auroc auprc precision recall <<< "$result"
            echo "Human,$beta,$seed,$N1,$N2,$ALPHA,$auroc,$auprc,$precision,$recall,$log" >> "$SUMMARY"
            printf "  AUROC=%.4f AUPRC=%.4f Precision=%.4f Recall=%.4f\n" "$auroc" "$auprc" "$precision" "$recall"
        fi
        echo ""
    done
done

echo "============================================"
echo " All done. Logs: $OUT_DIR/"
echo " Summary: $SUMMARY"
echo "============================================"
