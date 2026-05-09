#!/bin/bash
# run_davis.sh — Davis, 单 seed（分布式用）
# Usage: bash scripts/run_davis.sh <seed>
#   seed: 100, 1000, 2000, 3000, 4000

set -e
SEED=${1:?Usage: bash run_davis.sh <seed>}
cd "$(dirname "$0")/.."

OUTPUT="${HAG_DTA_OUTPUT_ROOT:-/root/autodl-tmp/HAG-DTA-runs}"
mkdir -p "$OUTPUT/logs"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$OUTPUT/logs/davis_s${SEED}.log"; }

log "========== Davis seed=$SEED =========="

# HAG-DTA GIN
log "START HAG-DTA GIN"
python -u -c "
import config.training as ct
ct.SEEDS = [$SEED]
import sys
sys.argv = ['training_davis_kiba.py', '0', '0']
exec(open('training_davis_kiba.py').read())
" >> "$OUTPUT/logs/davis_s${SEED}.log" 2>&1
log "DONE  HAG-DTA GIN"

# GraphDTA GCN
log "START GraphDTA GCN"
python -u -c "
import config.training as ct
ct.SEEDS = [$SEED]
import sys
sys.argv = ['training_graphdta.py', '0', '0']
exec(open('training_graphdta.py').read())
" >> "$OUTPUT/logs/davis_s${SEED}.log" 2>&1
log "DONE  GraphDTA GCN"

# GraphDTA GAT
log "START GraphDTA GAT"
python -u -c "
import config.training as ct
ct.SEEDS = [$SEED]
import sys
sys.argv = ['training_graphdta.py', '0', '1']
exec(open('training_graphdta.py').read())
" >> "$OUTPUT/logs/davis_s${SEED}.log" 2>&1
log "DONE  GraphDTA GAT"

# GraphDTA GIN
log "START GraphDTA GIN"
python -u -c "
import config.training as ct
ct.SEEDS = [$SEED]
import sys
sys.argv = ['training_graphdta.py', '0', '2']
exec(open('training_graphdta.py').read())
" >> "$OUTPUT/logs/davis_s${SEED}.log" 2>&1
log "DONE  GraphDTA GIN"

# GraphDTA SAGE
log "START GraphDTA SAGE"
python -u -c "
import config.training as ct
ct.SEEDS = [$SEED]
import sys
sys.argv = ['training_graphdta.py', '0', '3']
exec(open('training_graphdta.py').read())
" >> "$OUTPUT/logs/davis_s${SEED}.log" 2>&1
log "DONE  GraphDTA SAGE"

log "========== Davis seed=$SEED COMPLETE =========="
