#!/usr/bin/env bash
# Prints the summary tables for all P2-P5 designs on both datasets. See paper_designs/README.md section 7.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${WAVESPIKE_DATA_ROOT:?set WAVESPIKE_DATA_ROOT to your data folder first}"
cd "$HERE"; V=x_p2_dot,x_p4_sep,x_stf_ssa,x_stf_swsa,x_stf_fatm
echo "=== Zenodo across days ==="
python -I code/bio_run.py --dataset zenodo_bio --summary --variants "$V"
echo; echo "=== Seeing Red across sittings ==="
BIO_CHANNEL=luma python -I code/bio_run.py --dataset seeing_red --train-stride 0.5 --summary --variants "$V"
