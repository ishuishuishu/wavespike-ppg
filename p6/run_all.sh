#!/usr/bin/env bash
# Runs every missing P6 job one after another (3 families x 2 datasets = 6 jobs, 24 runs). Safe to rerun: finished runs are skipped.
# Usage: bash run_all.sh [threads=12]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for F in haar db2 db8; do
  for DS in zenodo seeing_red; do bash "$HERE/run_family.sh" "$F" "$DS" "${1:-12}"; done
done
echo "ALL DONE $(date)"
