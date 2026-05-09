#!/bin/bash
# sensitivity_n1n2_all.sh - run n1/n2 sensitivity on all datasets.
#
# Usage:
#   bash scripts/sensitivity_n1n2_all.sh
#
# Optional:
#   SEED=1000 bash scripts/sensitivity_n1n2_all.sh

set -e

cd "$(dirname "$0")/.."

echo "============================================"
echo " n1/n2 Sensitivity - all datasets"
echo " seed=${SEED:-100}"
echo "============================================"

bash scripts/sensitivity_n1n2_davis.sh
bash scripts/sensitivity_n1n2_kiba.sh
bash scripts/sensitivity_n1n2_human.sh
bash scripts/sensitivity_n1n2_celegans.sh

echo "============================================"
echo " All dataset sensitivity runs complete."
echo "============================================"
