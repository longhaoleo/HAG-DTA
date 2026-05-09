#!/bin/bash
# run_human_full.sh — Human 全量 (5 seeds, 单机串行)
# Usage: bash scripts/run_human_full.sh

set -e
cd "$(dirname "$0")/.."

echo "============================================"
echo " Human — full run (5 seeds)"
echo "============================================"

for seed in 100 1000 2000 3000 4000; do
    bash scripts/run_human.sh $seed
done

echo "============================================"
echo " Human COMPLETE"
echo "============================================"
