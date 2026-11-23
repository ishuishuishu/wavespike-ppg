#!/usr/bin/env bash
# Runs the 4 missing runs (folds 1,2 x seeds 0,1) of one P6 wavelet family on one dataset and saves the best run.
# See p6/README.md. Usage: bash run_family.sh <haar|db2|db8> <zenodo|seeing_red> [threads=12]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
F="${1:?family: haar, db2 or db8}"; DS="${2:?dataset: zenodo or seeing_red}"; THREADS="${3:-12}"
case "$F" in haar|db2|db8) ;; *) echo "family must be haar, db2 or db8"; exit 1;; esac
V="x_p6_$F"
: "${WAVESPIKE_DATA_ROOT:?set WAVESPIKE_DATA_ROOT to your data folder first (README section 4)}"
case "$DS" in
  zenodo)     test -f "$WAVESPIKE_DATA_ROOT/processed/zenodo_bio_50hz_10s.npz" || { echo "Zenodo data not prepared (README 4.3)"; exit 1; }
              ARGS=(--dataset zenodo_bio) ;;
  seeing_red) test -f "$WAVESPIKE_DATA_ROOT/raw/seeing_red/sr_rgb50.npz" || { echo "Seeing Red data not prepared (README 4.2)"; exit 1; }
              export BIO_CHANNEL=luma; ARGS=(--dataset seeing_red --train-stride 0.5) ;;
  *) echo "dataset must be zenodo or seeing_red"; exit 1 ;;
esac
mkdir -p "$HERE/logs"; cd "$HERE"; LOG="logs/${V}_${DS}.log"
echo "start $V $DS $(date)  commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)  threads $THREADS" | tee -a "$LOG"
python -I code/bio_run.py "${ARGS[@]}" --variants "$V" --folds 1,2 --seeds 0,1 --threads "$THREADS" \
    --save-model "$WAVESPIKE_DATA_ROOT/models/p6" --save-best-run --save-last-only 2>&1 | tee -a "$LOG"
echo "end $V $DS $(date)" | tee -a "$LOG"
