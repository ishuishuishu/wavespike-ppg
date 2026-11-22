#!/usr/bin/env bash
# Runs the 21 ablation runs still missing, one after another. See ablation/README.md section 6.
# Usage: bash run_missing.sh [threads=12]        Safe to rerun: finished runs are skipped.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THREADS="${1:-12}"
: "${WAVESPIKE_DATA_ROOT:?set WAVESPIKE_DATA_ROOT to your data folder first (README section 5.1)}"
test -f "$WAVESPIKE_DATA_ROOT/processed/zenodo_bio_50hz_10s.npz" || { echo "Zenodo data not prepared (README 5.3)"; exit 1; }
test -f "$WAVESPIKE_DATA_ROOT/raw/seeing_red/sr_rgb50.npz" || { echo "Seeing Red data not prepared (README 5.2)"; exit 1; }
mkdir -p "$HERE/logs"; cd "$HERE"
run() {  # run <log name> <bio_run.py arguments...>
  local log="logs/$1.log"; shift
  echo "start $(date)  commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)  threads $THREADS  args: $*" | tee -a "$log"
  python -I code/bio_run.py "$@" --threads "$THREADS" 2>&1 | tee -a "$log"
  echo "end $(date)" | tee -a "$log"
}
# Job 1: Seeing Red, fully learnable wavelet, no augmentation, seed 1 only (seed 0 is done)          3 runs
BIO_CHANNEL=luma run job1_seeing_red_x_tau_amp_seed1 --dataset seeing_red --train-stride 0.5 --variants x_tau_amp --folds 0,1,2 --seeds 1
# Job 2: Seeing Red, partly learnable (one shared filter), no augmentation                          6 runs
BIO_CHANNEL=luma run job2_seeing_red_x_shared_tau_amp --dataset seeing_red --train-stride 0.5 --variants x_shared_tau_amp --folds 0,1,2 --seeds 0,1
# Job 3: Zenodo, no wavelet, with augmentation                                                      6 runs
run job3_zenodo_x_aug_raw_tau_amp --dataset zenodo_bio --variants x_aug_raw_tau_amp --folds 0,1,2 --seeds 0,1
# Job 4: Zenodo, fixed db4 wavelet, with augmentation                                               6 runs
run job4_zenodo_x_aug_fixed_tau_amp --dataset zenodo_bio --variants x_aug_fixed_tau_amp --folds 0,1,2 --seeds 0,1
echo "ALL DONE $(date)"
