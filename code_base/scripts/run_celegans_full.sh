#!/bin/bash
# run_celegans_full.sh — C.elegans 全量 (5 seeds, 单机串行)
# Usage: bash scripts/run_celegans_full.sh

set -e
cd "$(dirname "$0")/.."

echo "============================================"
echo " C.elegans — full run (5 seeds)"
echo "============================================"

for seed in 100 1000 2000 3000 4000; do
    bash scripts/run_celegans.sh $seed
done

echo "============================================"
echo " C.elegans COMPLETE"
echo "============================================"
