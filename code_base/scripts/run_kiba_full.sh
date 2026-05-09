#!/bin/bash
# run_kiba_full.sh - KIBA full seed run.
# Usage: bash scripts/run_kiba_full.sh [models]

set -e
cd "$(dirname "$0")/.."

MODELS=${1:-gin}

echo "============================================"
echo " KIBA - full seed run (models=$MODELS)"
echo " KIBA can be slow; distributed single-seed runs are often easier."
echo "============================================"

for seed in 100 1000 2000 3000 4000; do
    bash scripts/run_kiba.sh "$seed" "$MODELS"
done

echo "============================================"
echo " KIBA COMPLETE"
echo "============================================"
