# Step 11 results: WaveSpike-PPG biometrics (corroborate, match or beat, Raspberry Pi export)

Date: 2026-11-23. Machine: Mac M2 (`snn/.venv`, Python 3.12.15, torch 2.6.0, NumPy 2.5.2), teammate repo commit `d2c7fe0`.
Protocol: `STEP11_PREDECLARED_PROTOCOL.md` (errata E1–E3; hashes in `results/step11_wavespike/protocol_hash_before_runs.txt`).
Deviations and notes: `STEP11_DEVIATIONS_AND_NOTES.md` (N1–N12).
Every number below comes from `results/step11_wavespike/step11_mac_summary.json` (recomputed from saved probabilities), `results/step11_wavespike/pi_bundle/parity_summary.json`, or (for the two descriptive B1 comparisons) `results/step11_wavespike/step11_mac_descriptive_extra.json`.

## Verdicts

| Question | Answer |
|---|---|
| Teammate's SNN (`x_tau_amp`) reproduced? | **Close, descriptive only.** 5-window EER 16.70 % vs their 15.62 % (+1.08 pp). Our data has 20 people, theirs 17 (E1), so this cannot be a strict replication. |
| Teammate's CNN benchmark (`cwt_cnn_amp`) reproduced? | **Close, descriptive only.** 12.06 % vs their 13.98 % (−1.92 pp; ours is better). |
| Selected SNN (by validation) | `x_tau_amp`: validation 5-window EER 14.80 % vs 15.16 % for B1 `x_tau_amp_more`. |
| Does the SNN match or beat the CNN? | **No: does not match.** The SNN's EER is 4.63 pp higher (95 % CI 2.87 to 6.40), and it wins 0/6 paired runs. |
| Project goal (FAR < 3 % and FRR < 5 % at the validation 2 %-FAR threshold) | **Not met** in any run (0/6). FRR is about 56 %. |
| Raspberry Pi export, predeclared parity | **FAIL, by one impostor trial.** The FAR count at tau is 117 (PyTorch) vs 118 (NumPy) out of 4,560; every other parity criterion passes. |
| NumPy port correct? (E3, supplementary) | **Yes.** In float64 the two agree to 5.7e-14 on all 1,200 test windows. The float32 differences are rounding. |

## Part A: their code and protocol, on our data (Mac, 3 folds × 2 seeds, last checkpoint)

| Model | Params | Test EER, 1 / 3 / 5 windows | Test top-1 (5) | Macro-F1 (5) | FAR / FRR (5) | Val EER (5) | Minutes per run |
|---|---|---|---|---|---|---|---|
| `x_tau_amp` (SNN) | 6,448 | 22.44 / 17.43 / **16.70 ± 1.93** | 47.6 % | 42.7 % | 2.31 / 56.0 % | 14.80 % | 99.7 |
| `cwt_cnn_amp` (CNN) | 24,860 | 19.90 / 14.01 / **12.06 ± 1.99** | 60.6 % | 56.6 % | 2.14 / 42.8 % | 9.92 % | 14.6 |

Teammate's reported values (17 people, 6 runs): `x_tau_amp` 5-window EER 15.62 %, top-1 49.2 %, 1-window EER 22.65 %. `cwt_cnn_amp` 5-window EER 13.98 %, top-1 61.4 %, 1-window EER 20.35 %.
The aggregator's predeclared labels would be "corroborated" (SNN, |Δ| ≤ 1.5 pp) and "partly corroborated" (CNN, |Δ| ≤ 3 pp). Under E1 they are **descriptive only**, because the cohorts differ.

Per-run 5-window test EER (%):

| Run | f0 ts0 | f0 ts1 | f1 ts0 | f1 ts1 | f2 ts0 | f2 ts1 |
|---|---|---|---|---|---|---|
| `x_tau_amp` | 18.75 | 18.33 | 15.53 | 13.75 | 16.01 | 17.81 |
| `x_tau_amp_more` (B1) | 17.92 | 17.94 | 14.91 | 13.33 | 17.04 | 15.81 |
| `cwt_cnn_amp` | 14.47 | 14.69 | 11.25 | 11.16 | 10.42 | 10.37 |

## Part B: more training data, and the benchmark comparison

- **B1 `x_tau_amp_more`** (the same SNN trained on 5× more training windows): test 5-window EER 16.16 ± 1.83 %, top-1 48.1 %, FAR/FRR 2.18/57.2 %, validation EER 15.16 %, 103.8 min per run.
- **Selection** (predeclared, on validation only): `x_tau_amp` 14.80 % < B1 15.16 %, so `x_tau_amp` is selected. The 0.36 pp margin is far larger than the recompute deviation (≤ 0.04 pp, N11).
- **Descriptive only** (test set; did not decide anything): B1 − `x_tau_amp` = −0.54 pp (95 % CI −1.56 to 0.48), with B1 lower in 5/6 runs. More data may help the SNN a little on test, but validation did not show it, and the CI includes 0.
- **Benchmark verdict** (selected SNN vs our `cwt_cnn_amp`, paired by fold and seed): mean Δ EER +4.63 pp (CI 2.87 to 6.40), with SNN lower in 0/6 runs. Per pair: +4.28, +3.64, +4.28, +2.59, +5.59, +7.43 pp. The predeclared rule says "beats" needs 6/6 wins and "matches" needs |Δ| ≤ 1 pp, so the result is **does not match**. Top-1 is 13.0 pp lower (CI −22.7 to −3.3). For reference, B1 vs the CNN is also +4.10 pp (CI 2.39 to 5.80), 0/6 wins.
- Against the teammate's *reported* CNN (13.98 %), the selected SNN is +2.72 pp.
- Dropped on compute grounds (E2): B2 `cwt_cnn_amp_more` and B3 `x_h128_tau_amp_more`.

## Part C: Raspberry Pi bundle (`results/step11_wavespike/pi_bundle/`)

Exported run: `x_tau_amp` fold 0, seed 1. It was chosen by the lower stored validation 5-window EER: 14.583 % vs 14.605 % for seed 0. Recomputed from the rounded probabilities, the two seeds tie (N11 e).
Test result of this run (PyTorch, 5 windows): EER 18.33 %, top-1 46.7 %, FAR 2.57 %, FRR 52.9 %, tau 0.2350.

| Predeclared parity criterion (full fold-0 test set, 1,200 windows) | Result | Pass? |
|---|---|---|
| 0 window-argmax mismatches | 0 | yes |
| \|Δ logit\| ≤ 1e-4 except threshold-adjacent windows | 5 windows > 1e-4, all threshold-adjacent (118 adjacent in total); max 0.37 | yes |
| Identical 5-window decisions | 0 / 240 differ | yes |
| Identical FAR / FRR counts at tau | FAR 117 vs **118**, FRR 127 vs 127 | **no** |
| **Overall (as predeclared)** | | **FAIL** |

Supplementary (E3): float64 PyTorch vs float64 NumPy max |Δ logit| 5.7e-14, with 0 mismatches (`IMPLEMENTATION_F64 PASS`). PyTorch float32 vs its own float64 differs by more than 1e-4 on 32 windows. So float32 rounding alone moves this recurrent spiking model by more than the parity tolerance, and one impostor score lands on the other side of tau.
Other checks: the checkpoint reproduces the run's saved test probabilities (max |Δp| 5.4e-7); our kernel vs the teammate's `edge/rpi_infer.py` gives max |Δ logit| 1.9e-6 on 50 windows; model file 35,012 bytes, SHA-256 `0d55bd44…213f`.
Pi self-check run on the Mac: 40/40 reference decisions reproduced (max |Δp| 2.3e-3), 55 ms per 5-window decision, about 9.9 ms per window. **No Raspberry Pi latency or energy is claimed.** These numbers come from the Mac; run `run_pi_selfcheck.sh` on the Pi.
Compatibility with the teammate's `edge/pi_bench.py` (N12): their `pi_bench.py` crashes with their own kernel at d2c7fe0 (`classify_window` has no `stats` argument). The bundle now carries it byte-identical, plus a small `rpi_infer.py` adapter (amplitude head included) and `run_pi_bench.sh`. It runs end to end on the Mac (8.3 ms per window). `energy_manual.py` measures energy per 5-window decision with a display-only USB meter (`PI_BENCH.md`).

## What this means

- On our 20-person version of the data, the teammate's numbers come out close for both models, and the gap between them is real. The CNN is better by about 4.6 pp EER in every paired run. The SNN uses about 26 % of the CNN's parameters (6,448 vs 24,860).
- Five times more training windows did not change the selection. On test it gave a small, not significant improvement (−0.54 pp).
- Neither model is near the project goal of FRR < 5 % at FAR < 3 %. Rejecting the right person about half the time is the main weakness of this approach on this data.
- The NumPy kernel runs the SNN correctly. The predeclared bit-for-bit parity fails on one impostor trial because of float32 rounding. Report it like that, not as a pass.

## Caveats

- Cohort: 20 people (Zenodo v1 files 001–020) vs the teammate's 17 (E1, N3). Part A is descriptive.
- Interim test results were seen before the final steps: NAS (N5) and the two exporter dry runs (N9, N10). Selection was by validation as predeclared, so these did not affect it.
- The aggregation tolerance was corrected after the queue's check failed (N11). The correction only lets FRR and top-1 differ by one decision; all deviations are recorded.
- The NAS runs (`results/step11_wavespike/nas_partial_2026-11-22/`) are a separate partial CPU replication and do not enter any verdict (E2).
- 6 runs per model (3 folds × 2 seeds), so the CIs are wide. The CNN's advantage holds in every pair.

## Files

- Summary: `results/step11_wavespike/step11_mac_summary.json`; descriptive B1 comparisons: `results/step11_wavespike/step11_mac_descriptive_extra.json`
- Logs: `results/step11_wavespike/mac_logs/` (`queue.out`, `A_cwt_cnn_amp.log`, `A_x_tau_amp.log`, `B1.log`, `aggregate.log` (failed check), `aggregate_rerun.log`, `export_parity.log`)
- Run files: `capstone/wavespike_data_mac/results/bio_zenodo_bio_w10/`, `.../step11_more_w10/`. Models: `capstone/wavespike_data_mac/models/step11/`
- Pi bundle: `results/step11_wavespike/pi_bundle/` (`snn_bio_model.npz`, `step11_rpi_infer.py`, `samples.npz`, `parity_summary.json`, `README.md`, `run_pi_selfcheck.sh`; added by `step11_add_pi_bench.py`: `pi_bench.py`, `rpi_infer.py`, `run_pi_bench.sh`, `bench_windows.npy`, `energy_manual.py`, `PI_BENCH.md`)
