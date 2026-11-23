#!/usr/bin/env bash
# Step 11 on the Mac (protocol erratum E2): waits for the lean SNN handoff (Task A) to finish, then
# W1: cwt_cnn_amp x6 -> x_tau_amp x6 (saved models) | W2: x_tau_amp_more x6 ; then aggregate, select, export + parity.
set -uo pipefail
CAP=${CAP:?set CAP to the folder that holds snn/.venv, metrics/, wavespike_data_mac/ and external/wavespike-ppg}
export WAVESPIKE_DATA_ROOT=$CAP/wavespike_data_mac WAVESPIKE_REPO=$CAP/external/wavespike-ppg
P=$CAP/snn/.venv/bin/python; M=$CAP/metrics; L=$M/results/step11_wavespike/mac_logs; R=$WAVESPIKE_REPO
LEAN=$R/lean_snn/logs
echo "queue waiting for lean SNN Task A $(date)"
until grep -qs "Task A end" $LEAN/task_a.log || { [ -f $LEAN/task_a.log ] && ! pgrep -f "run_task_[ab].sh" > /dev/null; }; do sleep 60; done
grep -q "Task A end" $LEAN/task_a.log || echo "WARNING: Task A ended without its end line (see $LEAN/task_a.log)"
echo "phase2 start $(date)  commit $(git -C $R rev-parse --short HEAD)"
( cd $R && $P -I biometrics/bio_run.py --dataset zenodo_bio --variants cwt_cnn_amp --folds 0,1,2 --seeds 0,1 --threads 4 > $L/A_cwt_cnn_amp.log 2>&1
  cd $R && $P -I biometrics/bio_run.py --dataset zenodo_bio --variants x_tau_amp --folds 0,1,2 --seeds 0,1 --threads 4 \
        --save-model $WAVESPIKE_DATA_ROOT/models/step11 > $L/A_x_tau_amp.log 2>&1
  echo "W1_DONE $(date)" ) &
( cd $M && $P -I step11_run.py --variants x_tau_amp_more --folds 0,1,2 --seeds 0,1 --threads 4 > $L/B1.log 2>&1
  echo "W2_DONE $(date)" ) &
wait
echo "training done $(date)"
cd $M && $P -I step11_aggregate.py results/step11_wavespike/teammate_reference_metrics.json results/step11_wavespike/step11_mac_summary.json > $L/aggregate.log 2>&1
echo "aggregate exit $?"
RUN=$($P -I - <<'PY'
import json, os, glob
s = json.load(open("results/step11_wavespike/step11_mac_summary.json")); sel = s["selection"]["selected"]
root = os.environ["WAVESPIKE_DATA_ROOT"] + "/results/" + ("step11_more_w10" if sel.endswith("_more") else "bio_zenodo_bio_w10")
c = {p: json.load(open(p))["last"]["metrics"][2]["eer_val"] for p in sorted(glob.glob(f"{root}/{sel}_f0_ts*.json"))}
print(min(c, key=c.get))
PY
)
echo "export run: $RUN"
cd $M && $P -I step11_export_parity.py "$RUN" $M/results/step11_wavespike/pi_bundle > $L/export_parity.log 2>&1
echo "export exit $? | $(grep PARITY $L/export_parity.log)"
echo "STEP11_MAC_DONE $(date)"
