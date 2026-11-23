# Step 11 replication: WaveSpike-PPG biometrics, rerun independently

An independent rerun of this repository's Zenodo biometric results, done 2026-11-22/23 on a separate machine with a protocol fixed before any neural result was seen.

**Questions it answers:**
1. Do `x_tau_amp` (our SNN) and `cwt_cnn_amp` (the CNN benchmark) come out as reported?
2. Does the SNN, or the SNN trained on 5× more training windows, match or beat the CNN?
3. Can the SNN be exported to NumPy for the Raspberry Pi with identical decisions?

> **Cohort warning: 20 people, not 17.** These runs used a Zenodo cache built with `biometrics/prepare_zenodo.py` from record 23597128 files `001`–`020`. That gives **20 subjects / 8,220 windows**.
>
> This repository's results use **17 subjects / 6,000 windows**: 002, 003, 005, 006, 007, 009, 013, 014, 015, 024, 030, 031, 033, 034, 036, 038, 039.
>
> The prep script is the same, so the raw files differ. Every comparison with the repository's numbers below is therefore **descriptive, not a strict replication**. Seeing Red was not part of Step 11.

## Results (3 folds × 2 seeds, last epoch, decision over 5 windows)

| Model | Test EER | Top-1 | FAR / FRR at the validation 2 %-FAR threshold | Repository (17 people) |
|---|---|---|---|---|
| `cwt_cnn_amp` (CNN) | **12.06 ± 1.99 %** | 60.6 % | 2.14 / 42.8 % | 13.98 % |
| `x_tau_amp` (SNN) | **16.70 ± 1.93 %** | 47.6 % | 2.31 / 56.0 % | 15.62 % |
| `x_tau_amp_more` (SNN, 5× training windows) | 16.16 ± 1.83 % | 48.1 % | 2.18 / 57.2 % | n/a |

- **Match or beat:** **no.** The SNN picked on validation (`x_tau_amp`, val EER 14.80 % vs 15.16 %) is +4.63 pp EER behind the CNN (95 % CI 2.87 to 6.40). It is worse in all 6 paired runs.
- **More training data:** −0.54 pp on test (CI −1.56 to 0.48), not significant, and not better on validation.
- **Project goal** (FAR < 3 % and FRR < 5 %): not met in any run.
- **NumPy / Pi export** (`x_tau_amp` fold 0, seed 1):
  - All 240 five-window decisions are identical. The **predeclared parity check fails by one impostor trial** (FAR count 117 PyTorch vs 118 NumPy out of 4,560).
  - In float64 the NumPy port matches PyTorch to 5.7e-14, so the port is correct and the gap is float32 rounding.
  - No Pi timing or energy was measured.

Full tables, per-run numbers and caveats: `STEP11_WAVESPIKE_RESULTS.md`.

## Findings for this repository

1. **`edge/pi_bench.py` crashes with `edge/rpi_infer.py`** (seen at commit `d2c7fe0`).
   - `pi_bench.py` calls `net.classify_window(x, stats=st)`, but `WaveSpikeNP.classify_window(self, x)` has no `stats` argument. It stops with `TypeError` at the op-count step.
   - `code/step11_pi_bench_adapter.py` works around it without changing `edge/`.
2. **`pi_bench.py --energy-from` uses `np.trapz`,** which recent NumPy no longer has (absent in 2.5.2). Use `np.trapezoid`.
3. **The Zenodo cohort depends on which raw files are downloaded.** See the warning above; which files or record version give the 17 is still open.

## What is in this folder

| Path | What |
|---|---|
| `STEP11_WAVESPIKE_RESULTS.md` | Results write-up. Every number is traced to the JSON files in `summary/` |
| `STEP11_PREDECLARED_PROTOCOL.md` | Protocol fixed before the runs, with dated errata E1–E3 appended (never edited) |
| `protocol_hash_before_runs.txt` | SHA-256 of the protocol at each stage. The last line matches this copy |
| `STEP11_DEVIATIONS_AND_NOTES.md` | Everything that differed from the plan, N1–N12 |
| `summary/step11_mac_summary.json` | All endpoints, recomputed from saved probabilities and cross-checked against each run's own JSON |
| `summary/step11_mac_descriptive_extra.json` | The two descriptive paired comparisons for `x_tau_amp_more` |
| `summary/teammate_reference_metrics.json` | The repository numbers the replication was compared against |
| `summary/pi_parity_summary.json` | PyTorch vs NumPy parity for the exported model |
| `code/` | The scripts that produced all of the above (below) |

**Not included** (git-ignored formats, or too large): the 18 per-run result JSONs (about 2 MB each, with saved probabilities), the 6 saved `x_tau_amp` models, the Pi bundle's `.npz` model and samples, the processed data and the logs. Ask the person who ran Step 11 for any of them.

**Paths inside the documents** (`results/step11_wavespike/...`, `capstone/wavespike_data_mac/...`) are from the original workspace. In this folder they correspond to `summary/` and `code/`.

## Code

| Script | Role |
|---|---|
| `step11_build_zenodo_more.py` | Rebuilds the Zenodo cache with extra **training** windows for `x_tau_amp_more`. It replays `prepare_zenodo.py` exactly (`--check` asserts the original windows match array for array) |
| `step11_check_valtest.py` | Checks that validation and test windows are identical with and without the extra training windows |
| `step11_run.py` | `bio_run.py`'s loop for the `_more` variants. The only change is that extra windows go to training only |
| `mac_step11_queue.sh` | The run queue used. `CAP` must point at the workspace (it was a fixed local path) |
| `step11_aggregate.py` | Recomputes every endpoint, cross-checks it, selects on validation, applies the predeclared verdicts |
| `step11_export_parity.py` | NumPy export, float32 parity and float64 port check, Pi bundle |
| `step11_rpi_infer.py` | NumPy inference: the `edge/rpi_infer.py` kernel plus the amplitude head, 5-window decisions, frozen threshold |
| `step11_add_pi_bench.py`, `step11_pi_bench_adapter.py`, `step11_energy_manual.py` | Make `edge/pi_bench.py` run on the bundle unchanged; energy per decision with a display-only USB meter |

**Environment:** Python 3.12.15, PyTorch 2.6.0, NumPy 2.5.2, CPU only (Apple M2). Repository code at commit `d2c7fe0` (`biometrics/`, `notebook/cells/`), unchanged.

**To rerun:**
1. Set `WAVESPIKE_DATA_ROOT` and `WAVESPIKE_REPO`.
2. Prepare Zenodo with `biometrics/prepare_zenodo.py`.
3. `python -I step11_build_zenodo_more.py $WAVESPIKE_DATA_ROOT --check ...`. It imports `prepare_zenodo` from a fixed path at the top of the file; adjust that line to your checkout.
4. Run the training commands in `mac_step11_queue.sh`.
5. `python -I step11_aggregate.py summary/teammate_reference_metrics.json <out.json>`.
6. `python -I step11_export_parity.py <run json> <bundle dir>`.

Each SNN run took about 100 min on 4 threads; each CNN run about 15 min.
