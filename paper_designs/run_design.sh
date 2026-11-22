#!/usr/bin/env bash
# Runs one design on one dataset: 3 folds x 2 seeds. See paper_designs/README.md.
# Usage: bash run_design.sh <variant> <zenodo|seeing_red> [threads=12]
#   variants: x_p2_dot  x_p4_sep  x_stf_ssa  x_stf_swsa  x_stf_fatm
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V="${1:?variant, e.g. x_p2_dot}"; DS="${2:?dataset: zenodo or seeing_red}"; THREADS="${3:-12}"
case "$V" in x_p2_dot|x_p4_sep|x_stf_ssa|x_stf_swsa|x_stf_fatm) ;; *) echo "unknown variant $V"; exit 1;; esac
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
python -I code/bio_run.py "${ARGS[@]}" --variants "$V" --folds 0,1,2 --seeds 0,1 --threads "$THREADS" 2>&1 | tee -a "$LOG"
echo "end $V $DS $(date)" | tee -a "$LOG"
