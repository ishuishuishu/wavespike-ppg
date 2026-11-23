#!/usr/bin/env bash
# Prints the summary tables for your P6 runs on both datasets. See p6/README.md section 7.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${WAVESPIKE_DATA_ROOT:?set WAVESPIKE_DATA_ROOT to your data folder first}"
cd "$HERE"; V=x_p6_haar,x_p6_db2,x_p6_db8
echo "=== Zenodo across days ==="
python -I code/bio_run.py --dataset zenodo_bio --summary --variants "$V"
echo; echo "=== Seeing Red across sittings ==="
BIO_CHANNEL=luma python -I code/bio_run.py --dataset seeing_red --train-stride 0.5 --summary --variants "$V"
