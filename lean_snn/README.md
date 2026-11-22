# Lean SNN: remaining runs (handoff)

This folder is everything a teammate needs to run the two remaining **lean SNN** experiments on their own server. Read it top to bottom once before starting. It takes about 1.5 hours of computer time in total, plus the data download.

| Task | What | Runs | Time (12 CPU threads) |
|---|---|---|---|
| **A** (required) | Lean SNN on **Seeing Red**, no augmentation, seed 1 | 3 | about 20 min |
| **B** (required) | Lean SNN **with augmentation** on **Zenodo** across days | 6 | about 1 hour |

Both tasks **save the trained models** as well as the results, so the models can be run on a Raspberry Pi later.

## Checklist: the whole job, in order

Each step links to its section. Tick them off as you go.

1. [ ] Clone the repo, create the Python environment, set `WAVESPIKE_DATA_ROOT` (§4.1).
2. [ ] Download and prepare Seeing Red; check it says **14 users** (§4.2).
3. [ ] Download and prepare Zenodo; check it says **17 subjects** (§4.3).
4. [ ] Run both smoke tests; check there's no `Traceback` (§4.4).
5. [ ] Run Task A: `bash run_task_a.sh` (§5).
6. [ ] Run Task B: `bash run_task_b.sh` (§6).
7. [ ] Print the results: `bash check_results.sh` (§8).
8. [ ] Check the models: `bash check_models.sh`, with 9 × `runs OK` (§9).
9. [ ] Send back everything in the Deliverables table (§9).

Task A and Task B are independent; run them one after the other, not at the same time.

## Deliverables (what to send back)

| # | Deliverable | Where it is | Count |
|---|---|---|---|
| 1 | Task A result files `x_lean_tau_amp_f{0,1,2}_ts1.json` | `$WAVESPIKE_DATA_ROOT/results/bio_seeing_red_luma_w10_trs0.5/` | 3 |
| 2 | Task B result files `x_aug_lean_tau_amp_f{0,1,2}_ts{0,1}.json` | `$WAVESPIKE_DATA_ROOT/results/bio_zenodo_bio_w10/` | 6 |
| 3 | Task A trained models `bio_seeing_red_luma_w10_trs0.5__x_lean_tau_amp_f{0,1,2}_ts1.pt` | `$WAVESPIKE_DATA_ROOT/models/lean_snn/` | 3 |
| 4 | Task B trained models `bio_zenodo_bio_w10__x_aug_lean_tau_amp_f{0,1,2}_ts{0,1}.pt` | `$WAVESPIKE_DATA_ROOT/models/lean_snn/` | 6 |
| 5 | Model check: output of `bash check_models.sh` (every model must print `runs OK`) | your terminal | 1 |
| 6 | Logs `task_a.log`, `task_b.log` | `logs/` | 2 |
| 7 | Results summary: output of `bash check_results.sh` | your terminal | 1 |
| 8 | Git commit hash you ran, plus Python, PyTorch and NumPy versions | see section 9 | 1 |
| 9 | Notes on anything unexpected (tracebacks, wrong person counts, odd numbers) | a short message | – |

**Done means:** 9 result files, 9 model files, 9 × `runs OK`, both logs, and the summary. Send them to the project lead directly (a zip is fine; the models total under 1 MB). Don't commit them to the repository or post them publicly.

Files in this folder:

| File | Use |
|---|---|
| `README.md` | This guide |
| `run_task_a.sh` | Runs Task A and saves its models |
| `run_task_b.sh` | Runs Task B and saves its models |
| `check_results.sh` | Prints the summary tables for both tasks |
| `check_models.sh` | Reloads every saved model and runs it once |
| `requirements.txt` | Python package versions used for the original runs |
| `code/bio_run.py` | Training, evaluation and summaries (the run scripts call it) |
| `code/cells/` | The model and training code that `bio_run.py` runs |
| `code/prepare_seeing_red.py`, `code/prepare_zenodo.py` | Turn the downloaded data into PPG windows |
| `code/load_model.py` | Rebuilds one saved model and runs it (used by `check_models.sh`; the starting point for the Pi) |
| `code/spiking_wavelets.py` | Imported by `bio_run.py`; not used by the lean SNN itself |
| `code/SOURCE.md` | Where each code file was copied from |

**This folder is self-contained.** Everything needed is inside it, so it runs on its own: you can copy just this folder to your server. Don't edit anything in `code/`. Run every command in this guide **from inside this folder** (`cd lean_snn`).

---

## 1. What the lean SNN is

The lean SNN is **our** model, not a reproduction of a published paper. It is our main spiking network (`x_tau_amp`) with two changes:

- **It drops two dead wavelet bands (`drop_bands = [0, 1]`).**
  - The learnable wavelet transform still splits the 50 Hz PPG into 6 bands: d1 to d5 plus the approximation.
  - Bands d1 (12.5–25 Hz) and d2 (6.25–12.5 Hz) are thrown away after the transform. In earlier runs their encoder neurons almost never fired (rate 0.001), so they carried no information.
  - The network keeps d3, d4, d5 and the approximation, all below 6.25 Hz.
- **The spiking part runs at half rate (`snn_decim = 2`).**
  - Every kept band is below 6.25 Hz, so the spiking network can take every 2nd sample (25 Hz) without aliasing.
  - The neuron leak factors are initialised as β₀², so time constants in seconds stay the same.

It **does** use wavelets. The rest is identical to the main SNN:
- learnable wavelet filters, one per level;
- learnable hard threshold;
- per-neuron time constants (`tau_het`);
- amplitude features fused into the readout (`amp_fusion`).

The two variants, defined in `code/bio_run.py`:

| Name | Settings |
|---|---|
| `x_lean_tau_amp` | 150 epochs, patience 150 (no early stopping), 1,600 class-balanced training windows per epoch, `tau_het`, `amp_fusion`, `drop_bands [0, 1]`, `snn_decim 2` |
| `x_aug_lean_tau_amp` | Same, plus `augment: True` (random amplitude and time changes to the training windows only) |

Size: about 6k parameters (5,766 with Seeing Red's 14 people; 5,973 with Zenodo's 17).

**Do not change any setting, default or line of code.** If something looks wrong, stop and ask; don't fix it.

## 2. Saved models

**There is no earlier lean model to continue from.** Our earlier biometric runs saved only result files (predictions and metrics), not weights. The only checkpoints on the original machine (`lean_tas_*.pt`) are from the atrial-fibrillation task: 2 classes and different data, so they can't be used here.

This doesn't block anything:
- Each run is a separate training from scratch for one (fold, seed) pair: 150 epochs, and the **last epoch** is the one reported.
- There is no half-trained lean run to resume. "Continuing" means running the missing (fold, seed) pairs below.

**Your runs save their models.** `bio_run.py` has an option, `--save-model DIR`, which is off by default and is used by both run scripts. After each run it writes one `.pt` file, about 60 KB, to `$WAVESPIKE_DATA_ROOT/models/lean_snn/`. Each file holds:
- `last_state`: the weights after the last epoch. **This is the model we report**, and the one for the Pi.
- `best_state`: the weights at the lowest validation loss, kept for reference.
- `cfg`: every model setting.
- `identities`: the person IDs, in class order.
- `amp_norm`: the mean and standard deviation used to scale the two amplitude features, computed on that fold's training data. The Pi needs these.
- `input`: a description of the expected input: a 10 s window at 50 Hz, z-scored per window, with the 2 amplitude features appended.
- The variant, dataset, fold, seed and best epoch.

The option changes nothing else: training, results and result files are exactly as without it.

To check a model rebuilds and runs:

```bash
python -I code/load_model.py $WAVESPIKE_DATA_ROOT/models/lean_snn/<file>.pt
```

It prints the variant, fold, seed, parameter count and `runs OK`. `bash check_models.sh` does this for every saved model.

**For the Pi (later, not part of this handoff).** `load_model.py` needs PyTorch, which runs on a 64-bit Raspberry Pi OS. The NumPy-only Pi exporter in the main project (not in this folder) doesn't yet handle the amplitude-feature term these models use; extending it is a separate task.

## 3. How the tests work (so your numbers are comparable)

- **Task.** Identify a person from 10 s of PPG. Each person is one class.
- **Split by session.** Training, validation and test windows come from **different sessions**, so the model has to recognise people on a day or sitting it hasn't seen:
  - Zenodo: a session is a recording day.
  - Seeing Red: a session is a sitting.
- **Folds and seeds.**
  - 3 folds (0, 1, 2) rotate which sessions are held out for validation and test.
  - 2 training seeds (0, 1) change only the network's starting weights and the sampling.
  - So a full set is 3 × 2 = 6 runs.
- **Windows.** 10 s windows (`--win 10`, the default). Validation and test windows start every 2 s (`--stride 2`, the default).
  - Seeing Red only: training windows start every 0.5 s (`--train-stride 0.5`).
  - Zenodo: training windows use the default 2 s.
- **Metrics.** Each one is reported for decisions on 1, 3 and 5 consecutive windows:
  - top-1 identification accuracy;
  - macro-F1;
  - EER (equal error rate, lower is better);
  - FAR/FRR at the threshold picked on validation data.

  We quote the **last epoch** and **5 windows**.

## 4. Setup (once)

### 4.1 Code and Python

1. Get this folder. Either clone the repository and go into it:

   ```bash
   git clone https://github.com/ishuishuishu/wavespike-ppg.git
   cd wavespike-ppg/lean_snn
   git log --oneline -1        # note this hash; you'll send it back
   ```

   or copy just the `lean_snn` folder to your server and `cd` into it. In that case, send back the name of the file or zip you received instead of a commit hash.

2. Create a Python 3.13 (or newer) environment with CPU PyTorch and the packages in `requirements.txt`. A GPU isn't needed or used. The original runs used Python 3.14 and PyTorch 2.15.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

   If an exact version in `requirements.txt` isn't available for your Python, install the nearest one and note it in your report.

3. Make a data folder **outside** this folder and point `WAVESPIKE_DATA_ROOT` at it:

   ```bash
   mkdir -p ~/wavespike_data/raw ~/wavespike_data/processed
   export WAVESPIKE_DATA_ROOT=~/wavespike_data
   ```

**In every new shell**, before running anything, `cd` into the `lean_snn` folder and run these two lines again (or add them to your `~/.bashrc`):

```bash
source .venv/bin/activate
export WAVESPIKE_DATA_ROOT=~/wavespike_data
```

The run scripts stop with a clear message if `WAVESPIKE_DATA_ROOT` isn't set or the data isn't prepared.

Download the data on the server itself.

**Your data will match ours exactly.** Both preparation scripts are deterministic: same input files, same windows, in the same order (fixed random seed). The folds are built from those windows, so your fold 0 is our fold 0. Use the full downloads, with no files left out.

### 4.2 Seeing Red data (needed for Task A)

- **Source.** The dataset for Lovisotto, Turner, Eberz and Martinovic, *Seeing Red: PPG Biometrics Using Smartphone Cameras* (CVPR Workshops 2020), on the Oxford Research Archive (ora.ox.ac.uk; search for "Seeing Red"). Open access, ODC-PDDL licence.
- **What it is.** Fingertip-on-phone-camera videos of 14 people over several sittings. The videos zip is about 590 MB.

1. Download the videos zip from that record and unzip it. You need a folder with **one subfolder per person**, each holding `.mp4` files. Example: `~/seeing_red/videos/`.
2. Convert the videos to PPG (mean red/luma intensity per frame, inverted, resampled to 50 Hz):

   ```bash
   mkdir -p $WAVESPIKE_DATA_ROOT/raw/seeing_red
   python -I code/prepare_seeing_red.py ~/seeing_red/videos $WAVESPIKE_DATA_ROOT/raw/seeing_red/sr_rgb50.npz 12
   ```

   The last number is the number of parallel workers. Check the printed line: it must say **14 users**.

### 4.3 Zenodo data (needed for Task B)

- **Source.** Zenodo record 23597128: long-term wrist PPG with ECG. CC BY-NC-SA 4.0 licence. <https://zenodo.org/records/23597128>
- **What you need.** Only the `NNN_PPG.mat` files. The `_ECG.mat` files aren't needed for biometrics.

1. Download every `*_PPG.mat` file into `$WAVESPIKE_DATA_ROOT/raw/zenodo_23597128/`.
2. Build the identity windows (10 s, up to 60 per day, quality-gated, 100 → 50 Hz):

   ```bash
   python -I code/prepare_zenodo.py $WAVESPIKE_DATA_ROOT
   ```

   It prints one line per subject and ends with `saved .../processed/zenodo_bio_50hz_10s.npz: ... windows, 17 subjects`. It must say **17 subjects**.

### 4.4 Smoke test (2 minutes; writes no result files)

```bash
BIO_CHANNEL=luma python -I code/bio_run.py --dataset seeing_red --variants x_lean_tau_amp --folds 0 --seeds 1 --train-stride 0.5 --smoke
python -I code/bio_run.py --dataset zenodo_bio --variants x_aug_lean_tau_amp --folds 0 --seeds 0 --smoke
```

Each must print a line with `identities 14` (Seeing Red) or `identities 17` (Zenodo) and finish without a `Traceback`.

---

## 5. Task A (required): Seeing Red, lean SNN, seed 1

**Why.** On Seeing Red, `x_lean_tau_amp` has only 3 of its 6 runs (seed 0). We need seed 1 to report it with the same 6 repeats as every other model.

| Setting | Value |
|---|---|
| Dataset | `seeing_red`, luma channel (`BIO_CHANNEL=luma`) |
| Variant | `x_lean_tau_amp` |
| Folds | 0, 1, 2 |
| Seed | **1 only** (seed 0 is done; don't rerun it) |
| Training stride | `--train-stride 0.5` |
| Runs | 3, about 6 min each on 12 threads |

Run:

```bash
bash run_task_a.sh            # optional: a thread count, e.g. bash run_task_a.sh 8
```

That is exactly:

```bash
BIO_CHANNEL=luma python -I code/bio_run.py --dataset seeing_red --variants x_lean_tau_amp \
    --folds 0,1,2 --seeds 1 --train-stride 0.5 --threads 12 \
    --save-model $WAVESPIKE_DATA_ROOT/models/lean_snn
```

**Output.**
- Three result files in `$WAVESPIKE_DATA_ROOT/results/bio_seeing_red_luma_w10_trs0.5/`: `x_lean_tau_amp_f0_ts1.json`, `x_lean_tau_amp_f1_ts1.json`, `x_lean_tau_amp_f2_ts1.json`.
- Three model files in `$WAVESPIKE_DATA_ROOT/models/lean_snn/`: `bio_seeing_red_luma_w10_trs0.5__x_lean_tau_amp_f{0,1,2}_ts1.pt`. Each run prints `saved model ...` when its file is written.
- The console log goes to `logs/task_a.log`.

**What to expect.** On the original machine, seed 0 (last epoch, 5 windows, mean of 3 runs) gave **EER 25.4%** (range 20.0–30.4%) and top-1 37.6%. Seed 1 should land in a similar range, but not identical: the seed differs, and the thread count changes results slightly.

## 6. Task B (required): Zenodo, lean SNN with augmentation

**Why.** On Zenodo across days, we have the lean SNN without augmentation and the full SNN with augmentation, but not the lean SNN **with** augmentation. This fills that gap, so "lean vs full" can be compared with augmentation on both datasets. (On Seeing Red, `x_aug_lean_tau_amp` is already done.)

| Setting | Value |
|---|---|
| Dataset | `zenodo_bio` |
| Variant | `x_aug_lean_tau_amp` |
| Folds | 0, 1, 2 |
| Seeds | 0, 1 |
| Training stride | default (2 s); **do not** pass `--train-stride` |
| Runs | 6, about 10 min each on 12 threads |

Run:

```bash
bash run_task_b.sh            # optional thread count, as above
```

That is exactly:

```bash
python -I code/bio_run.py --dataset zenodo_bio --variants x_aug_lean_tau_amp \
    --folds 0,1,2 --seeds 0,1 --threads 12 \
    --save-model $WAVESPIKE_DATA_ROOT/models/lean_snn
```

**Output.**
- Six result files `x_aug_lean_tau_amp_f{0,1,2}_ts{0,1}.json` in `$WAVESPIKE_DATA_ROOT/results/bio_zenodo_bio_w10/`.
- Six model files `bio_zenodo_bio_w10__x_aug_lean_tau_amp_f{0,1,2}_ts{0,1}.pt` in `$WAVESPIKE_DATA_ROOT/models/lean_snn/`.
- The log goes to `logs/task_b.log`.

**What to expect.** Our runs on Zenodo for the nearby models (last epoch, 5 windows, mean of 6 runs):

| Model | EER | Top-1 |
|---|---|---|
| Lean SNN, no augmentation (`x_lean_tau_amp`) | 16.9% (15.2–18.6) | 46.9% |
| Full SNN, with augmentation (`x_aug_tau_amp`) | 16.0% (14.5–18.1) | 43.9% |
| Full SNN, no augmentation (`x_tau_amp`) | 15.6% (14.0–18.1) | 49.2% |

Task B is a new model/dataset pair, so there is no "correct" number. A mean EER far outside about 13–22% would be worth reporting as unexpected.

## 7. If a run is interrupted

- Just run the same script again.
- Each finished (fold, seed) writes its own result file and model file, and the script skips any run whose result file already exists.
- If a run has a result file but no model file, tell the project lead. Don't delete the result file to force a rerun.
- A run that was cut off midway starts again from epoch 0. There's no mid-run checkpoint.
- Don't delete or edit result files.

## 8. Check the results

```bash
bash check_results.sh
```

This prints the summary tables:
- Task A, Seeing Red: your 3 runs only.
- Task B, Zenodo: your 6 runs, next to whichever reference runs are in your results folder.

Read the rows where `n = 5`, the "EER% b/l" column, after the slash (last epoch).

## 9. Check the models and send everything back

1. Check every saved model rebuilds and runs:

   ```bash
   bash check_models.sh
   ```

   It must list 9 models, each ending in `runs OK`. The predicted identity it prints is for a synthetic pulse, so it means nothing; only `runs OK` matters.
2. Get your versions for deliverable 8:

   ```bash
   git log --oneline -1
   python -c "import sys, torch, numpy; print(sys.version, torch.__version__, numpy.__version__)"
   ```

3. Send everything in the **Deliverables** table at the top of this guide, unchanged. A zip of `results/bio_seeing_red_luma_w10_trs0.5/x_lean_tau_amp_*_ts1.json`, `results/bio_zenodo_bio_w10/x_aug_lean_tau_amp_*.json`, `models/lean_snn/` and `logs/` plus the two printouts covers it.

Report anything unexpected:
- a `Traceback` or warning;
- a person/subject count other than 14 (Seeing Red) or 17 (Zenodo);
- numbers far outside the ranges above;
- a model that doesn't print `runs OK`.

Send these to the project lead directly. Don't commit result files, models or logs to the repository, and don't post them publicly.

## 10. Project rules

- Open-access data only.
- PPG is the only model input.
- Don't change defaults, settings or code without asking first.
- Report only what your own runs produced.
- Run one job at a time. About 12 threads is fastest on a CPU.
