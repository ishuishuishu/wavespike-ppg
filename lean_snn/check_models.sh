#!/usr/bin/env bash
# Reloads every saved lean SNN model and runs it once. Each must print "runs OK". See lean_snn/README.md section 9.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WAVESPIKE_DATA_ROOT:?set WAVESPIKE_DATA_ROOT to your data folder first}"
cd "$REPO"
shopt -s nullglob
M=("$WAVESPIKE_DATA_ROOT"/models/lean_snn/*.pt)
echo "${#M[@]} model files in $WAVESPIKE_DATA_ROOT/models/lean_snn (expected 9)"
for f in "${M[@]}"; do
    echo "== $(basename "$f")"
    python -I lean_snn/load_model.py "$f" 2>&1 | tail -n 2
done
echo "checked ${#M[@]} files; look for 'runs OK' under each one"
