#!/bin/bash
# run_davis_full.sh — Davis 全量 (5 seeds × 5 models, 单机串行)
# Usage: bash scripts/run_davis_full.sh

set -e
cd "$(dirname "$0")/.."

echo "============================================"
echo " Davis — full run (5 seeds × 5 models)"
echo " 预计耗时：较长（串行 25 次训练）"
echo "============================================"

for seed in 100 1000 2000 3000 4000; do
    bash scripts/run_davis.sh $seed
done

echo "============================================"
echo " Davis COMPLETE"
echo "============================================"
