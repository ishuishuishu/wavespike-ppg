# P6 (wavelet-family comparison), rebuilt for PPG

This folder has everything needed to finish our PPG version of paper **P6** and save its trained models. It is self-contained: copy just this folder to a server and follow this guide top to bottom.

## Your job, in short

1. Prepare two open datasets (Seeing Red and Zenodo).
2. Train **3 wavelet families** (`haar`, `db2`, `db8`) on **both datasets**, for the **4 missing runs each** (folds 1 and 2 × seeds 0 and 1). That's **24 training runs**. One command, `bash run_all.sh`, does all of it, one run at a time, in about one day.
3. Each run writes a result file. For each family on each dataset, the best of its 4 runs (chosen on validation data) is also saved as a trained model for the Raspberry Pi: **6 models**.
4. Run two checks, then send back the files listed below.

Fold 0 of these three families, and all 6 runs of `sym4` and `coif1`, are **already done** by us (§3). Don't rerun them.

## Deliverables (what to send back)

| # | Deliverable | Where | Count |
|---|---|---|---|
| 1 | Zenodo result files: `x_p6_haar`, `x_p6_db2`, `x_p6_db8`, each as `_f1_ts0`, `_f1_ts1`, `_f2_ts0`, `_f2_ts1.json` | `$WAVESPIKE_DATA_ROOT/results/bio_zenodo_bio_w10/` | 12 |
| 2 | Seeing Red result files, same names | `$WAVESPIKE_DATA_ROOT/results/bio_seeing_red_luma_w10_trs0.5/` | 12 |
| 3 | Trained models, one per family per dataset (exact names in §8) | `$WAVESPIKE_DATA_ROOT/models/p6/` | 6 |
| 4 | Printout of `bash check_models.sh`: 6 models, each with `runs OK` | your terminal | 1 |
| 5 | Printout of `bash check_results.sh`: the summary tables | your terminal | 1 |
| 6 | Logs, one per family per dataset | `logs/` | 6 |
| 7 | Commit hash (or the name of the zip you received), plus Python, PyTorch and NumPy versions | commands in §9 | 1 |
| 8 | A short note on anything unexpected (tracebacks, wrong person counts, odd numbers, a model without `runs OK`), or "nothing unexpected" | a message | 1 |

**Done means:** 24 result files, 6 model files, 6 × `runs OK`, the summary, 6 logs, your versions and the note. §9 has one command that zips everything up.

## What P6 is, and what "recreate" means here

**P6: Aghabarar et al. 2023, Journal of AI and Data Mining (JAIDM).** It turns a signal into spikes in two stages:
1. a **fixed discrete wavelet transform** (DWT) as preprocessing, splitting the signal into frequency bands;
2. a **leaky integrate-and-fire (LIF) encoder** that turns each band into spikes, followed by a spiking network.

Its main experiment compares **wavelet families** (which fixed filter the DWT uses). P6 is not a PPG paper, so we can't rerun its experiment. Instead its design is rebuilt for 1-D PPG and run inside **our** biometric pipeline: same data, folds, seeds, training recipe and metrics. Only the wavelet changes.

- Results from this folder are **our runs of a design rebuilt from the paper's text**. They are not the paper's own numbers and must not be reported as such.
- Every choice the paper leaves open is written in the docstring at the top of `code/spiking_wavelets.py`.

## The design

| Part | P6 (as rebuilt) | Our own SNN (for comparison) |
|---|---|---|
| Wavelet | **Fixed** filters from PyWavelets, family = `haar`, `db2`, `db4`, `db8`, `sym4` or `coif1`. Undecimated (stationary) transform, 5 levels → 6 bands. **Nothing in the wavelet is learned** | Learnable: one 8-tap filter per level, started from db4, trained end to end |
| Encoder | Constant-current LIF, the same as our encoder: a "+" and a "−" neuron per band, 12 in total | Same |
| Spiking network | Our recurrent SNN: 64 LIF neurons, time constants spread 0.2–4 s | Same |
| Output | Leaky readout, one neuron per person, plus two amplitude features (signal level, pulse size) | Same |

So P6 is **our SNN with a fixed wavelet**, one run per family. The question it answers: *does the choice of fixed wavelet family matter, and does a learnable wavelet beat all of them?*

| Variant | Wavelet family | Filter length |
|---|---|---|
| `x_p6_haar` | Haar | 2 taps |
| `x_p6_db2` | Daubechies-2 | 4 taps |
| `x_p6_db8` | Daubechies-8 | 16 taps |
| `x_p6_sym4` | Symlet-4 (done) | 8 taps |
| `x_p6_coif1` | Coiflet-1 (done) | 6 taps |

(db4 is our existing "fixed db4" SNN, `x_fixed_tau_amp`, so P6 doesn't need its own db4 run.)

## How the code is organised

There's **no separate program for P6**. It lives in the shared code:

| What | Where |
|---|---|
| The P6 wavelet (`class FixedSWT`) | `code/spiking_wavelets.py` |
| The P6 variants (`x_p6_*`) | `code/bio_run.py`, the loop starting `for _n, _pw in [("p6_haar", ...` |
| Our SNN, training loop and recipe | `code/cells/` |
| Data loading, folds, training, metrics, saving | `code/bio_run.py` |

- **Why shared:** the comparison only means something if everything except the wavelet is identical.
- The files also contain P1, P2, P4, P5 and P7. Those aren't part of this job; ignore them.

## Checklist

1. [ ] Set up Python and the data folder (§4.1).
2. [ ] Download and prepare Seeing Red; check it says **14 users** (§4.2).
3. [ ] Download and prepare Zenodo; check it says **17 subjects** (§4.3).
4. [ ] Run the smoke test; check there's no `Traceback` (§5).
5. [ ] Run all jobs: `bash run_all.sh` (§6).
6. [ ] Print the results: `bash check_results.sh` (§7).
7. [ ] Check the saved models: `bash check_models.sh`, with 6 × `runs OK` (§8).
8. [ ] Zip and send back the deliverables (§9).

Run every command **from inside this folder** (`cd p6`). Run one job at a time.

## Files in this folder

| File | Use |
|---|---|
| `README.md` | This guide |
| `run_family.sh` | Runs the 4 missing runs of one family on one dataset and saves its best run |
| `run_all.sh` | Runs all 3 families on both datasets, one after another (6 jobs, 24 runs) |
| `check_results.sh` | Prints the summary tables for your runs |
| `check_models.sh` | Reloads every saved model and runs it once |
| `p6_runs_so_far.csv` | Every P6 run we have already finished, one row per run (§3) |
| `code/load_model.py` | Rebuilds one saved model and runs it (the starting point for the Pi) |
| `requirements.txt` | Python package versions used for our runs |
| `code/spiking_wavelets.py` | The P6 design, with every gap-filling choice documented |
| `code/bio_run.py` | Training, evaluation and summaries |
| `code/cells/` | Our SNN, training loop and recipe |
| `code/prepare_seeing_red.py`, `code/prepare_zenodo.py` | Turn the downloaded data into PPG windows |
| `code/SOURCE.md` | Where each code file was copied from |

**Do not change any setting, default or line of code.** If something looks wrong, stop and ask.

---

## 1. How the tests work

- **Task.** Identify a person from 10 s of PPG. Each person is one class.
- **Split by session.** Training, validation and test windows come from different sessions, so the model has to recognise people on a day or sitting it hasn't seen:
  - Zenodo: a session is a recording day.
  - Seeing Red: a session is a sitting.
- **Repeats.** 3 folds (0, 1, 2) × 2 training seeds (0, 1) = 6 runs per family per dataset. You run folds 1 and 2; we ran fold 0.
- **Windows.** 10 s windows. Validation and test windows start every 2 s.
  - Seeing Red only: training windows start every 0.5 s (`--train-stride 0.5`).
  - Zenodo: training windows use the default 2 s.
- **Training recipe** (the same as our SNN): 150 epochs, 1,600 class-balanced windows per epoch, no early stopping, and the **last epoch** is reported. No augmentation.
- **Metrics:** top-1 accuracy, macro-F1, EER (equal error rate, lower is better) and FAR/FRR, for decisions on 1, 3 and 5 windows. We quote the **last epoch at 5 windows**.

## 2. Datasets

| Dataset | Used for | Licence |
|---|---|---|
| Seeing Red (Lovisotto et al., CVPRW 2020) | Across sittings, 14 people | ODC-PDDL (open) |
| Zenodo 23597128 (long-term wrist PPG) | Across days, 17 people | Check the record's licence |

## 3. What has already been run (our runs)

Last epoch, 5 windows. Every individual run is in `p6_runs_so_far.csv`.

| Variant | Runs done | Zenodo EER | Seeing Red EER | Status |
|---|---|---|---|---|
| `x_p6_haar` | fold 0 only (2 of 6) | 17.6% | 32.4% | **you run folds 1, 2** |
| `x_p6_db2` | fold 0 only (2 of 6) | 17.2% | 30.2% | **you run folds 1, 2** |
| `x_p6_db8` | fold 0 only (2 of 6) | 19.6% | 27.7% | **you run folds 1, 2** |
| `x_p6_sym4` | all 6 | 16.5% | 26.1% | done |
| `x_p6_coif1` | all 6 | 16.2% | 26.1% | done |

Fold-0 numbers are higher than 6-run means on Zenodo because fold 0 is the hardest fold there (sym4 and coif1 are 17.6–19.7% on fold 0 and about 15% on folds 1 and 2), so don't read much into the fold-0 column alone. Seeing Red varies a lot between folds (13–37% for sym4).

For comparison, our own models on the same data (6-run means unless noted):

| Our model | Zenodo EER | Seeing Red EER |
|---|---|---|
| Our SNN, fully learnable wavelet (`x_tau_amp`) | 15.6% | 28.7% (3 of 6 runs) |
| Our SNN, fixed db4 wavelet (`x_fixed_tau_amp`) | 16.1% | 24.7% |
| Our SNN, no wavelet (`x_raw_tau_amp`) | 20.2% | 29.5% |

**We combine** your 4 runs per family with our 2 fold-0 runs into 6-run means. You don't need to do that.

## 4. Setup (once)

### 4.1 Python and data folder

```bash
cd p6
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
mkdir -p ~/wavespike_data/raw ~/wavespike_data/processed
export WAVESPIKE_DATA_ROOT=~/wavespike_data
```

- Python 3.13 or newer, CPU only. Our runs used Python 3.14 and PyTorch 2.15.
- If an exact version in `requirements.txt` isn't available, install the nearest one and note it in your report.
- **In every new shell**, `cd p6`, then run `source .venv/bin/activate` and `export WAVESPIKE_DATA_ROOT=~/wavespike_data` again.
- If you already prepared the data for `lean_snn`, `ablation` or `paper_designs`, point `WAVESPIKE_DATA_ROOT` at the same folder and skip §4.2–4.3.

**Your data will match ours exactly.** Both preparation scripts are deterministic, so your folds are our folds, and your fold 1 and 2 runs fit together with our fold 0 runs.

### 4.2 Seeing Red

1. On the Oxford Research Archive (ora.ox.ac.uk), find the dataset for Lovisotto, Turner, Eberz and Martinovic, *Seeing Red: PPG Biometrics Using Smartphone Cameras* (CVPR Workshops 2020). Download the videos zip (about 590 MB) and unzip it. You need one subfolder per person, each with `.mp4` files.
2. Convert it:

   ```bash
   mkdir -p $WAVESPIKE_DATA_ROOT/raw/seeing_red
   python -I code/prepare_seeing_red.py <videos_folder> $WAVESPIKE_DATA_ROOT/raw/seeing_red/sr_rgb50.npz 12
   ```

   It must print **14 users**.

### 4.3 Zenodo

1. From <https://zenodo.org/records/23597128>, download every `*_PPG.mat` file into `$WAVESPIKE_DATA_ROOT/raw/zenodo_23597128/`. The `_ECG.mat` files aren't needed.
2. Build the windows:

   ```bash
   python -I code/prepare_zenodo.py $WAVESPIKE_DATA_ROOT
   ```

   It must end with `... 6000 windows, 17 subjects`.

## 5. Smoke test (a few minutes; writes no result files)

Each family trains for 2 epochs on fold 1 only:

```bash
for V in x_p6_haar x_p6_db2 x_p6_db8; do
  python -I code/bio_run.py --dataset zenodo_bio --variants $V --folds 1 --seeds 0 --smoke
done
BIO_CHANNEL=luma python -I code/bio_run.py --dataset seeing_red --variants x_p6_haar --folds 1 --seeds 0 --train-stride 0.5 --smoke
```

Each must print `identities 17` (or `identities 14` for Seeing Red), then one result line, with no `Traceback`.

## 6. Run the jobs

**Everything, in order** (6 jobs: each family on Zenodo, then on Seeing Red):

```bash
bash run_all.sh            # optional thread count: bash run_all.sh 8
```

**Or one job at a time**, in any order:

```bash
bash run_family.sh haar zenodo
bash run_family.sh haar seeing_red
# ... and the same for db2 and db8
```

That is exactly, for example:

```bash
python -I code/bio_run.py --dataset zenodo_bio --variants x_p6_haar --folds 1,2 --seeds 0,1 --threads 12 \
    --save-model $WAVESPIKE_DATA_ROOT/models/p6 --save-best-run --save-last-only
BIO_CHANNEL=luma python -I code/bio_run.py --dataset seeing_red --train-stride 0.5 --variants x_p6_haar --folds 1,2 --seeds 0,1 --threads 12 \
    --save-model $WAVESPIKE_DATA_ROOT/models/p6 --save-best-run --save-last-only
```

If you type a command yourself, keep the three `--save-...` options so the model is saved.

**Time.** On a quiet machine with about 12 threads, finished P6 runs took about **35–70 min each**. 24 runs ≈ **one day** in total. db8 (16-tap filter) may be a little slower.

**Output:**
- One result file per run, `<variant>_f<fold>_ts<seed>.json`, in:
  - `$WAVESPIKE_DATA_ROOT/results/bio_zenodo_bio_w10/` (Zenodo);
  - `$WAVESPIKE_DATA_ROOT/results/bio_seeing_red_luma_w10_trs0.5/` (Seeing Red).
- One trained model per job, the job's **best run**: `<results folder name>__<variant>_bestval.pt` (about 30–50 KB), in `$WAVESPIKE_DATA_ROOT/models/p6/`.
  - After each run, its validation EER is compared with the saved model's. If it's lower, the file is replaced and the run prints `saved model ... (val EER x%)`. Otherwise it prints `model not kept: ...`.
- A log per job in `logs/`.

**If interrupted:** run the same command again. Finished runs (those with a result file) are skipped; an interrupted run starts again from epoch 0. If a run has a result file but no model file, tell the project lead; don't delete the result file to force a rerun. Don't delete or edit result files.

## 7. Check the results

```bash
bash check_results.sh
```

- It prints one table per dataset, with a block of 3 rows (n = 1, 3, 5 windows) for each family. It shows **4 runs** per family (yours); that's expected.
- Read the rows where `n = 5`, the `EER% b/l` column, after the slash (last epoch). `top1 b/l` after the slash is the matching top-1 accuracy.
- Expect roughly 14–20% EER on Zenodo and 13–38% on Seeing Red (Seeing Red varies a lot by fold). Report anything far outside that as unexpected.

## 8. Saved models (for the Raspberry Pi later)

One model per family per dataset, **6 in total**: the best of your 4 runs, last-epoch weights.

- **How "best" is picked:** lowest **validation** EER (last epoch, 5 windows). Test data is never used to choose.
- **The results don't change:** the paper still reports the mean of all 6 runs.

The 6 files you should end up with:

```text
bio_zenodo_bio_w10__x_p6_haar_bestval.pt
bio_zenodo_bio_w10__x_p6_db2_bestval.pt
bio_zenodo_bio_w10__x_p6_db8_bestval.pt
bio_seeing_red_luma_w10_trs0.5__x_p6_haar_bestval.pt
bio_seeing_red_luma_w10_trs0.5__x_p6_db2_bestval.pt
bio_seeing_red_luma_w10_trs0.5__x_p6_db8_bestval.pt
```

Each `.pt` file holds:
- `last_state`: the weights after the last epoch. **This is the model we report**, and the one for the Pi. (`best_state` is left empty.)
- `cfg`: every model setting, including the wavelet family.
- `identities`: the person IDs, in class order.
- `amp_norm`: the mean and standard deviation used to scale the two amplitude features (computed on that fold's training data).
- The variant, dataset, and the fold and seed of the chosen run.
- `val_eer_last_n5`: the chosen run's validation EER.

Check every model rebuilds and runs:

```bash
bash check_models.sh
```

- It must list **6** models, each ending in `runs OK`.
- The predicted identity it prints is for a synthetic pulse and means nothing; only `runs OK` matters.
- To check one file: `python -I code/load_model.py $WAVESPIKE_DATA_ROOT/models/p6/<file>.pt`.

## 9. Package and send the deliverables

```bash
cd p6
bash check_models.sh  > check_models.txt  2>&1
bash check_results.sh > check_results.txt 2>&1
{ git log --oneline -1 2>/dev/null || echo "no git: <name of the zip you received>"
  python -c "import sys, torch, numpy; print(sys.version, torch.__version__, numpy.__version__)"; } > versions.txt
R=$WAVESPIKE_DATA_ROOT
zip -r p6_results.zip check_models.txt check_results.txt versions.txt logs \
    $R/results/bio_zenodo_bio_w10/x_p6_{haar,db2,db8}_f[12]_ts*.json \
    $R/results/bio_seeing_red_luma_w10_trs0.5/x_p6_{haar,db2,db8}_f[12]_ts*.json \
    $R/models/p6/*_bestval.pt
```

Before sending, check:
- `check_models.txt` lists **6** models, each with `runs OK`;
- `check_results.txt` shows **4 runs** for every family on both datasets;
- the zip has 24 `.json` files and 6 `.pt` files (`unzip -l p6_results.zip`).

Send the zip and your short note (deliverable 8) to the project lead directly. Don't commit it to the repository or post it publicly.

## 10. Project rules

- Open-access data only.
- PPG is the only model input.
- Don't change defaults, settings or code without asking first.
- Report only what your own runs produced, and label these as a design rebuilt from the paper's text, not the paper's results.
- Run one job at a time. About 12 threads is fastest on a CPU.
