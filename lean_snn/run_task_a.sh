#!/usr/bin/env bash
# Task A: lean SNN (x_lean_tau_amp) on Seeing Red across sittings, seed 1, folds 0-2. See lean_snn/README.md section 5.
# Usage: bash lean_snn/run_task_a.sh [threads=12]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THREADS="${1:-12}"
: "${WAVESPIKE_DATA_ROOT:?set WAVESPIKE_DATA_ROOT to your data folder first (README section 4.1)}"
test -f "$WAVESPIKE_DATA_ROOT/raw/seeing_red/sr_rgb50.npz" || { echo "missing $WAVESPIKE_DATA_ROOT/raw/seeing_red/sr_rgb50.npz (README section 4.2)"; exit 1; }
mkdir -p "$REPO/lean_snn/logs"
cd "$REPO"
export BIO_CHANNEL=luma
echo "Task A start $(date)  commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)  threads $THREADS" | tee -a lean_snn/logs/task_a.log
python -I biometrics/bio_run.py --dataset seeing_red --variants x_lean_tau_amp \
    --folds 0,1,2 --seeds 1 --train-stride 0.5 --threads "$THREADS" \
    --save-model "$WAVESPIKE_DATA_ROOT/models/lean_snn" 2>&1 | tee -a lean_snn/logs/task_a.log
echo "Task A end $(date)" | tee -a lean_snn/logs/task_a.log
