#!/usr/bin/env bash
# Prints the summary tables for Task A and Task B. See lean_snn/README.md section 8.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WAVESPIKE_DATA_ROOT:?set WAVESPIKE_DATA_ROOT to your data folder first}"
cd "$REPO"
echo "=== Task A: Seeing Red across sittings (x_lean_tau_amp) ==="
BIO_CHANNEL=luma python -I biometrics/bio_run.py --dataset seeing_red --train-stride 0.5 --summary --variants x_lean_tau_amp
echo
echo "=== Task B: Zenodo across days (x_aug_lean_tau_amp; other rows appear only if their result files exist) ==="
python -I biometrics/bio_run.py --dataset zenodo_bio --summary --variants x_aug_lean_tau_amp,x_lean_tau_amp,x_aug_tau_amp,x_tau_amp
