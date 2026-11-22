#!/usr/bin/env bash
# Task B: lean SNN with augmentation (x_aug_lean_tau_amp) on Zenodo across days, folds 0-2 x seeds 0-1. See lean_snn/README.md section 6.
# Usage: bash lean_snn/run_task_b.sh [threads=12]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THREADS="${1:-12}"
: "${WAVESPIKE_DATA_ROOT:?set WAVESPIKE_DATA_ROOT to your data folder first (README section 4.1)}"
test -f "$WAVESPIKE_DATA_ROOT/processed/zenodo_bio_50hz_10s.npz" || { echo "missing $WAVESPIKE_DATA_ROOT/processed/zenodo_bio_50hz_10s.npz (README section 4.3)"; exit 1; }
mkdir -p "$REPO/lean_snn/logs"
cd "$REPO"
echo "Task B start $(date)  commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)  threads $THREADS" | tee -a lean_snn/logs/task_b.log
python -I biometrics/bio_run.py --dataset zenodo_bio --variants x_aug_lean_tau_amp \
    --folds 0,1,2 --seeds 0,1 --threads "$THREADS" \
    --save-model "$WAVESPIKE_DATA_ROOT/models/lean_snn" 2>&1 | tee -a lean_snn/logs/task_b.log
echo "Task B end $(date)" | tee -a lean_snn/logs/task_b.log
