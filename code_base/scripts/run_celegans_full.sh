#!/bin/bash
# run_celegans_full.sh - C.elegans full seed run.
# Usage: bash scripts/run_celegans_full.sh [models]

set -e
cd "$(dirname "$0")/.."

MODELS=${1:-gin}

echo "============================================"
echo " C.elegans - full seed run (models=$MODELS)"
echo "============================================"

for seed in 100 1000 2000 3000 4000; do
    bash scripts/run_celegans.sh "$seed" "$MODELS"
done

echo "============================================"
echo " C.elegans COMPLETE"
echo "============================================"
