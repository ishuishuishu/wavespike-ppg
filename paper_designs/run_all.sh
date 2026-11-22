#!/usr/bin/env bash
# Runs every P2-P5 job one after another (5 designs x 2 datasets = 10 jobs, 60 runs). Safe to rerun: finished runs are skipped.
# Usage: bash run_all.sh [threads=12]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for V in x_p2_dot x_p4_sep x_stf_ssa x_stf_swsa x_stf_fatm; do
  for DS in zenodo seeing_red; do bash "$HERE/run_design.sh" "$V" "$DS" "${1:-12}"; done
done
echo "ALL DONE $(date)"
