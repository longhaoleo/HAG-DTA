#!/bin/bash
# run_human_full.sh - Human full seed run.
# Usage: bash scripts/run_human_full.sh [models]

set -e
cd "$(dirname "$0")/.."

MODELS=${1:-gin}

echo "============================================"
echo " Human - full seed run (models=$MODELS)"
echo "============================================"

for seed in 100 1000 2000 3000 4000; do
    bash scripts/run_human.sh "$seed" "$MODELS"
done

echo "============================================"
echo " Human COMPLETE"
echo "============================================"
