#!/bin/bash
# run_kiba_full.sh — KIBA 全量 (5 seeds × 5 models, 单机串行)
# Usage: bash scripts/run_kiba_full.sh

set -e
cd "$(dirname "$0")/.."

echo "============================================"
echo " KIBA — full run (5 seeds × 5 models)"
echo " 预计耗时：极长（KIBA 单个 seed ~30h）"
echo " 建议：分布式跑 run_kiba.sh <seed>"
echo "============================================"

for seed in 100 1000 2000 3000 4000; do
    bash scripts/run_kiba.sh $seed
done

echo "============================================"
echo " KIBA COMPLETE"
echo "============================================"
