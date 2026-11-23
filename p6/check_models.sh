#!/usr/bin/env bash
# Reloads every saved P6 model and runs it once. Each must print "runs OK". See p6/README.md section 8.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${WAVESPIKE_DATA_ROOT:?set WAVESPIKE_DATA_ROOT to your data folder first}"
cd "$HERE"
shopt -s nullglob
M=("$WAVESPIKE_DATA_ROOT"/models/p6/*.pt)
echo "${#M[@]} model files in $WAVESPIKE_DATA_ROOT/models/p6 (expected 6)"
for f in "${M[@]}"; do
    echo "== $(basename "$f")"
    python -I code/load_model.py "$f" 2>&1 | tail -n 2
done
echo "checked ${#M[@]} files; look for 'runs OK' under each one"
