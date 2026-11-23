# Step 11 protocol (predeclared): corroborate WaveSpike-PPG, match or beat its benchmark, export to the Pi

*Written 2026-11-22, before any Step 11 result exists. Nothing in it changes after results. Any change goes in a dated erratum appended at the end.*

## Goal
The user asked for three things, using the teammate's repository as the reference:
1. **Corroborate** the teammate's results.
2. **Train a model** that matches or beats their benchmarks.
3. **Make it ready for the Raspberry Pi.**

All training and evaluation run on the **NAS only**. Nothing runs on the Mac.

## Reference
- **Repository:** `ishuishuishu/wavespike-ppg` at commit `325e2220c25f530015a24e36535981009b1909e2`, cloned to `../external/wavespike-ppg` and copied to the NAS at `~/capstone_metrics_run/wavespike-ppg`. Its code is not modified. New code lives in separate files.
- **Reference numbers:** the embedded metrics of `docs/results.html` (SHA-256 `d577b8e4…0174`), extracted to `results/step11_wavespike/teammate_reference_metrics.json`. Their primary readout is used throughout:
  - **last-epoch checkpoint;**
  - **decision from 5 windows** (≈ 50 s of PPG);
  - **EER;**
  - mean over 3 folds × 2 training seeds.
- **Benchmark to match or beat** (their best model on the hard test, Zenodo wrist PPG, across days, 17 people): **`cwt_cnn_amp`**.
  - 5-window EER **13.98 %** and top-1 61.4 %;
  - 1-window EER 20.35 %.
  - Their SNN `x_tau_amp` scores 15.62 % and 49.2 % (1-window EER 22.65 %).

## Data
- **Zenodo record 23597128 (version 1, created 2026-10-23),** the record the teammate's README names. Licence: non-commercial (`other-nc`; the README says CC BY-NC-SA 4.0), so it is used for academic, non-commercial research only.
  - A newer version, 23921320, reassigns subject IDs, re-synchronises the PPG and drops 4 PPG files. **It is not used**, so the comparison stays like-for-like.
- **Assumption D1.** The teammate's handoff says 20 subjects were downloaded (19 usable for AF, 17 for biometrics). We assume these were **subjects 001–020**, the first 20 IDs.
  - **Test of the assumption, decided now:** `prepare_zenodo.py` is deterministic. Run unchanged on 001–020, it must give **17 subjects, 6,000 windows, and 1,020 test windows per fold** (17 × 60, read from their `n_decisions`).
  - If any count differs, the assumption is wrong. The corroboration is then reported as "cohort not reproduced" and nothing is adjusted to match.
- **Downloads:** PPG files 001–020 plus `LICENSE.txt` and `subject_info.xlsx` from record 23597128, each checked against Zenodo's MD5.

## Part A: corroboration (their code and protocol, unchanged)
- **Command:**
  ```
  bio_run.py --dataset zenodo_bio --folds 0,1,2 --seeds 0,1
  ```
  with the defaults: 10-s windows; session-disjoint folds where session = recording day, and fold 0 tests on each person's latest day.
- **Variants:**
  - `x_tau_amp`: their SNN;
  - `cwt_cnn_amp`: the benchmark;
  - `beat_svm_amp` and `dwt_svm_amp`: classical, deterministic; these check that the data is identical.
- **Verdicts** (5-window EER, last checkpoint, mean over runs, against the reference):
  - **Classical models** (deterministic, so the data must match):
    - **exact:** |Δ| ≤ 0.3 pp;
    - **not exact:** otherwise. Library versions differ (scikit-learn 1.9.0 here), so small differences are possible; any difference is reported.
  - **Neural models:**
    - **corroborated:** |Δ mean| ≤ 1.5 pp;
    - **partly corroborated:** 1.5–3 pp;
    - **not corroborated:** > 3 pp.

    CPU, thread count and library versions differ from the teammate's machine, so bitwise equality is not expected. Their 6-run SD is about 1.4–3.6 pp.
- **Also reported:** 1- and 3-window EER, top-1, macro-F1, FAR and FRR at their threshold (2 % FAR on validation), per-run values and minutes per run.

## Part B: match or beat the benchmark
**The one change, and why.** The teammate's cache keeps **at most 60 windows per person per day**. Each person has 5–8 recording days of ~24 h, so most usable signal is discarded. Enrolment data is the most direct lever across days.
- **Enlarged training pool:** for every (person, day) the original cache kept, continue the **same random candidate order** past window 60 and keep up to **300** windows that pass the same quality gate (≥ 0.35). The try limit is 40 × 300.
- **Original windows stay identical.** The first 60 accepted windows of each kept day are the teammate's windows, and this is verified array-for-array against Part A's cache. Days and subjects are exactly those of the original cache.
- **Training only.** Extra windows are used **only for training-day indices**. **Validation and test windows are exactly Part A's** (the extra windows are masked out of validation and test), so test sets are identical and pairs are directly comparable.

**Candidates** (predeclared; nothing else will be trained in Part B):

| ID | Model | Change from the teammate's recipe |
|---|---|---|
| B1 `x_tau_amp_more` | Their SNN, unchanged | Enlarged training pool |
| B2 `cwt_cnn_amp_more` | The benchmark, unchanged | Enlarged training pool (fairness control: the benchmark gets the same data) |
| B3 `x_h128_tau_amp_more` | Their SNN with 128 hidden neurons | Enlarged training pool (more data may let the bigger SNN help; on the original data it did not) |

- **Seeds and folds:** the same 3 folds × 2 seeds; same epochs, same per-epoch sampling (1,600 windows), same optimiser.
  - The SNN's training cost per epoch is unchanged.
  - The CWT-CNN iterates over its whole training set each epoch, so B2 costs about 5× more.
- **Runner:** `step11_run.py` (new). It reuses `bio_run.py`'s functions and main-loop logic, adding only the enlarged loader and the validation/test mask.
  - **Validated before use:** run on the *original* data it must reproduce Part A's `beat_svm_amp` probabilities exactly (max |Δp| ≤ 1e-6).
- **Selection** (validation only): the **deployable model** is the SNN (`x_tau_amp`, B1 or B3) with the **lowest mean validation 5-window EER** (last checkpoint). Test numbers are reported for every candidate.
- **Verdicts against the benchmark:** all on test, 5-window EER, paired by (fold, seed), using the teammate's all-repeats rule.
  - **beats:** lower EER than our Part A `cwt_cnn_amp` in **6/6** paired runs;
  - **matches:** |Δ mean| ≤ 1.0 pp and fewer than 6/6 wins;
  - **does not match:** otherwise.

  Also reported:
  - the same comparison against the reported 13.98 %, as a descriptive figure;
  - against B2, so we can see whether the gain is the data or the model;
  - our project goal (FAR < 3 %, FRR < 5 %). We expect it not to be met across days, and that will be stated.

## Part C: Raspberry Pi export
- **Which model:** the selected SNN variant's **fold-0 run** (strictly chronological: enrol on the past, test on the latest day), using the training seed with the lower **validation** 5-window EER.
- **Export** (new `step11_export.py`, NumPy `.npz`):
  - everything `rpi_infer.py` needs;
  - plus the amplitude-fusion head: the Linear(2 → K) weights and the training-fold mean and SD of [window mean, log window SD], computed from the **raw** window;
  - plus class names, configuration and source checkpoint SHA-256.
- **NumPy inference** (new `step11_rpi_infer.py`; the teammate's file is not modified):
  - their event-driven kernel, plus amplitude fusion;
  - 5-window aggregation (mean log-probability, as `bio_run.aggregate`);
  - the verification threshold τ frozen from validation (2 % FAR, their rule).
- **Parity on the full fold-0 test set:**
  - **0 window-argmax mismatches;**
  - max |Δ logit| ≤ 1e-4, except windows where a membrane potential sits within 1e-5 of threshold (counted and disclosed, as in step 6);
  - **identical** 5-window decisions and identical FAR/FRR counts at τ.
- **Pi bundle:** model, inference code, sample test windows with the PyTorch reference decisions, a self-check, and a run script compatible with the teammate's `edge/pi_bench.py`.
  - **No Pi numbers are claimed.** Latency here is NAS-only and labelled so.

## Environment changes (recorded now)
- NAS venv `~/capstone_metrics_run/.venv`: installed `h5py 3.12.1` (reads the `.mat` files) and `matplotlib 3.10.0` (imported by notebook cell 02). Everything else is unchanged: torch 2.6.0+cpu, numpy 2.5.2, scipy 1.18.1, scikit-learn 1.9.0, PyWavelets 1.8.0, pandas 3.0.5.
- The teammate's code expects Python ≥ 3.13; the NAS has 3.12.3. If anything fails for that reason, it is reported, not patched silently.

## Rules (unchanged from earlier steps)
- No tuning on test.
- Every endpoint is recomputed from the saved probabilities and checked against each run's JSON.
- Every number in the results doc is checked against the JSON.
- No historical result is overwritten.
- All pairs run on one machine (the NAS).
- Assumptions are recorded, not asked.

---

## Erratum E1 (2026-11-22, 12:53 IST; before any neural result)
- Assumption D1 failed: subjects 001–020 give **20** people (8,220 windows; 1,200 test windows per fold), not the teammate's 17. As predeclared, the cohort is reported as **not reproduced** and nothing is adjusted. Part A comparisons with the teammate's reported numbers are **descriptive only** (different cohort). Part B verdicts are paired within our own runs and are unaffected. Details: `STEP11_DEVIATIONS_AND_NOTES.md` N3.

## Erratum E2 (2026-11-23; after interim NAS results were seen, see N5/N7, before any Mac Step 11 run)
Written for **compute reasons only**: the NAS needed about 7.7 h per SNN run (about 4 days in all), and the user stopped it and moved the work to the Mac.
- **Machine.** All Step 11 runs that enter a verdict now run on **one machine, the Mac** (Apple M2; `snn/.venv`, Python 3.12.15, torch 2.6.0; teammate repo commit `d2c7fe0`, whose `cwt_cnn_probs` default recipe and SNN code paths are unchanged from `325e222`). The NAS results (classical 6/6, `cwt_cnn_amp` 6/6, `x_tau_amp` 1/6, `cwt_cnn_amp_more` 2/6) are kept as a separate, partial CPU replication and **do not enter any verdict**.
- **Runs (Mac):** `cwt_cnn_amp` 6, `x_tau_amp` 6 (via the teammate's `bio_run.py`, with its opt-in `--save-model`, so the run can be exported), and **B1** `x_tau_amp_more` 6 (via `step11_run.py`). Folds 0–2 × seeds 0–1, last checkpoint, as predeclared.
- **Dropped on compute grounds:** B2 `cwt_cnn_amp_more` and B3 `x_h128_tau_amp_more`. The selection is now among **`x_tau_amp` and B1** by mean validation 5-window EER. The benchmark is the Mac's `cwt_cnn_amp`. "Beats" means 6/6 paired wins, and "matches" means |mean Δ| ≤ 1 pp. Both are unchanged.
- **Part C** exports the selected SNN's fold-0 run: whichever seed has the lower validation 5-window EER. Parity criteria are unchanged.
- **Priority:** the teammate's lean SNN handoff (Tasks A and B) finishes first, on the same Mac, before any Step 11 run starts. Step 11 runs use 2 workers × 4 threads.

## Erratum E3 (2026-11-23, 05:14 IST; after an exporter dry run on a Part A run, before the selection and the Part C export)
Written after a **dry run** of `step11_export_parity.py` on the Mac's `x_tau_amp` fold 0 seed 0 run (the first Mac model saved; see N9). That run's float32 parity broke the predeclared rule: 1 window-argmax mismatch out of 1,200, and 3 windows with |Δ logit| > 1e-4 whose closest membrane was more than 1e-5 from threshold. Both sides were then rerun in float64 with the same weights. They agreed to 4.3e-14 on all 1,200 windows, with 0 mismatches. PyTorch float32 vs PyTorch float64 differed by more than 1e-4 on 50 windows, with 1 argmax change. So the difference is float32 rounding in a recurrent spiking net, not a port error.
- **The predeclared Part C parity rule is unchanged.** It is still applied to the final export and reported as PASS or FAIL as it comes out.
- **Added, reported separately and not replacing the rule:** an implementation-equivalence check, `IMPLEMENTATION_F64`. It passes if PyTorch and NumPy, both in float64 with the same weights, give max |Δ logit| ≤ 1e-9 and 0 argmax mismatches on the full fold-0 test set. Also reported, for scale: PyTorch float32 vs float64 (windows over 1e-4 and argmax changes), plus the existing decision-level counts (5-window argmax mismatches and FAR/FRR counts at tau).
- **Reading:** "PARITY FAIL | IMPLEMENTATION_F64 PASS" means the NumPy kernel is a correct port, and its float32 output differs from PyTorch's float32 output about as much as rounding alone moves the model. It is reported as exactly that. It is not called a parity pass.
