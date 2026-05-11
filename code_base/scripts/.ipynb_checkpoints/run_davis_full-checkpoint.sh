#!/bin/bash
# run_davis_full.sh - Davis full seed run.
# Usage: bash scripts/run_davis_full.sh [models]

set -e
cd "$(dirname "$0")/.."

MODELS=${1:-gin}

echo "============================================"
echo " Davis - full seed run (models=$MODELS)"
echo "============================================"

for seed in 100 1000 2000 3000 4000; do
    bash scripts/run_davis.sh "$seed" "$MODELS"
done

echo "============================================"
echo " Davis COMPLETE"
echo "============================================"
