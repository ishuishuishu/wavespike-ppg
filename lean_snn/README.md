# Lean SNN: remaining runs (handoff)

This folder is everything a teammate needs to run the two remaining **lean SNN** experiments on their own server. Read it top to bottom once before starting. It takes about 1.5 hours of computer time in total, plus the data download.

| Task | What | Runs | Time (12 CPU threads) |
|---|---|---|---|
| **A** (required) | Lean SNN on **Seeing Red**, no augmentation, seed 1 | 3 | about 20 min |
| **B** | Lean SNN **with augmentation** on **Zenodo** across days | 6 | about 1 hour |

Files in this folder:

| File | Use |
|---|---|
| `README.md` | This guide |
| `run_task_a.sh` | Runs Task A |
| `run_task_b.sh` | Runs Task B |
| `check_results.sh` | Prints the summary tables for both tasks |

The scripts call the shared project code (`biometrics/bio_run.py` and `notebook/cells/`), so keep this folder inside the repository and run the scripts from anywhere in it.

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

The two variants, defined in `biometrics/bio_run.py`:

| Name | Settings |
|---|---|
| `x_lean_tau_amp` | 150 epochs, patience 150 (no early stopping), 1,600 class-balanced training windows per epoch, `tau_het`, `amp_fusion`, `drop_bands [0, 1]`, `snn_decim 2` |
| `x_aug_lean_tau_amp` | Same, plus `augment: True` (random amplitude and time changes to the training windows only) |

Size: about 6k parameters (5,766 with Seeing Red's 14 people; 5,973 with Zenodo's 17).

**Do not change any setting, default or line of code.** If something looks wrong, stop and ask; don't fix it.

## 2. Is there a saved model to continue training?

**No.** The biometric runs save only a result file (predictions and metrics), not the trained weights. The only checkpoints on the original machine (`lean_tas_*.pt`) are from the atrial-fibrillation task: 2 classes and different data, so they can't be used here.

This doesn't block anything:
- Each run is a separate training from scratch for one (fold, seed) pair: 150 epochs, and the **last epoch** is the one reported.
- There is no half-trained lean run to resume. "Continuing" means running the missing (fold, seed) pairs below.

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

1. Clone the repository and check out the latest `main`:

   ```bash
   git clone https://github.com/ishuishuishu/wavespike-ppg.git
   cd wavespike-ppg
   git log --oneline -1        # note this hash; you'll send it back
   ```

2. Create a Python 3.13 (or newer) environment with CPU PyTorch and the packages below. A GPU isn't needed or used.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install numpy scipy scikit-learn pandas h5py PyWavelets opencv-python-headless
   ```

3. Make a data folder **outside** the repository and point `WAVESPIKE_DATA_ROOT` at it. Add the `export` line to every new shell:

   ```bash
   mkdir -p ~/wavespike_data/raw ~/wavespike_data/processed
   export WAVESPIKE_DATA_ROOT=~/wavespike_data
   ```

Download the data on the server itself.

### 4.2 Seeing Red data (needed for Task A)

- **Source.** Seeing Red (Lovisotto et al., CVPRW 2020), on the Oxford Research Archive. Open access, ODC-PDDL licence.
- **What it is.** Fingertip-on-phone-camera videos of 14 people over several sittings.

1. Download the videos zip from the Oxford Research Archive record for the Seeing Red dataset and unzip it. You need a folder with **one subfolder per person**, each holding `.mp4` files. Example: `~/seeing_red/videos/`.
2. Convert the videos to PPG (mean red/luma intensity per frame, inverted, resampled to 50 Hz):

   ```bash
   mkdir -p $WAVESPIKE_DATA_ROOT/raw/seeing_red
   python -I biometrics/prepare_seeing_red.py ~/seeing_red/videos $WAVESPIKE_DATA_ROOT/raw/seeing_red/sr_rgb50.npz 12
   ```

   The last number is the number of parallel workers. Check the printed line: it must say **14 users**.

### 4.3 Zenodo data (needed for Task B)

- **Source.** Zenodo record 23597128: long-term wrist PPG with ECG. CC BY-NC-SA 4.0 licence. <https://zenodo.org/records/23597128>
- **What you need.** Only the `NNN_PPG.mat` files. The `_ECG.mat` files aren't needed for biometrics.

1. Download every `*_PPG.mat` file into `$WAVESPIKE_DATA_ROOT/raw/zenodo_23597128/`.
2. Build the identity windows (10 s, up to 60 per day, quality-gated, 100 → 50 Hz):

   ```bash
   python -I biometrics/prepare_zenodo.py $WAVESPIKE_DATA_ROOT
   ```

   It prints one line per subject and ends with `saved .../processed/zenodo_bio_50hz_10s.npz: ... windows, 17 subjects`. It must say **17 subjects**.

### 4.4 Smoke test (2 minutes; writes no result files)

```bash
BIO_CHANNEL=luma python -I biometrics/bio_run.py --dataset seeing_red --variants x_lean_tau_amp --folds 0 --seeds 1 --train-stride 0.5 --smoke
python -I biometrics/bio_run.py --dataset zenodo_bio --variants x_aug_lean_tau_amp --folds 0 --seeds 0 --smoke
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
bash lean_snn/run_task_a.sh            # optional: a thread count, e.g. bash lean_snn/run_task_a.sh 8
```

That is exactly:

```bash
BIO_CHANNEL=luma python -I biometrics/bio_run.py --dataset seeing_red --variants x_lean_tau_amp \
    --folds 0,1,2 --seeds 1 --train-stride 0.5 --threads 12
```

**Output.** Three files in `$WAVESPIKE_DATA_ROOT/results/bio_seeing_red_luma_w10_trs0.5/`:
- `x_lean_tau_amp_f0_ts1.json`
- `x_lean_tau_amp_f1_ts1.json`
- `x_lean_tau_amp_f2_ts1.json`

The console log goes to `lean_snn/logs/task_a.log`.

**What to expect.** On the original machine, seed 0 (last epoch, 5 windows, mean of 3 runs) gave **EER 25.4%** (range 20.0–30.4%) and top-1 37.6%. Seed 1 should land in a similar range, but not identical: the seed differs, and the thread count changes results slightly.

## 6. Task B: Zenodo, lean SNN with augmentation

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
bash lean_snn/run_task_b.sh            # optional thread count, as above
```

That is exactly:

```bash
python -I biometrics/bio_run.py --dataset zenodo_bio --variants x_aug_lean_tau_amp \
    --folds 0,1,2 --seeds 0,1 --threads 12
```

**Output.** Six files `x_aug_lean_tau_amp_f{0,1,2}_ts{0,1}.json` in `$WAVESPIKE_DATA_ROOT/results/bio_zenodo_bio_w10/`. The log goes to `lean_snn/logs/task_b.log`.

**What to expect.** Our runs on Zenodo for the nearby models (last epoch, 5 windows, mean of 6 runs):

| Model | EER | Top-1 |
|---|---|---|
| Lean SNN, no augmentation (`x_lean_tau_amp`) | 16.9% (15.2–18.6) | 46.9% |
| Full SNN, with augmentation (`x_aug_tau_amp`) | 16.0% (14.5–18.1) | 43.9% |
| Full SNN, no augmentation (`x_tau_amp`) | 15.6% (14.0–18.1) | 49.2% |

Task B is a new model/dataset pair, so there is no "correct" number. A mean EER far outside about 13–22% would be worth reporting as unexpected.

## 7. If a run is interrupted

- Just run the same script again.
- Each finished (fold, seed) writes its own result file, and the script skips any run whose file already exists.
- A run that was cut off midway starts again from epoch 0. There's no mid-run checkpoint.
- Don't delete or edit result files.

## 8. Check the results

```bash
bash lean_snn/check_results.sh
```

This prints the summary tables:
- Task A, Seeing Red: your 3 runs only.
- Task B, Zenodo: your 6 runs, next to whichever reference runs are in your results folder.

Read the rows where `n = 5`, the "EER% b/l" column, after the slash (last epoch).

## 9. What to send back

1. The result files, unchanged:
   - Task A: `x_lean_tau_amp_f{0,1,2}_ts1.json`.
   - Task B: `x_aug_lean_tau_amp_f{0,1,2}_ts{0,1}.json`.
2. `lean_snn/logs/task_a.log` and `lean_snn/logs/task_b.log`.
3. The output of `bash lean_snn/check_results.sh`.
4. The git commit hash you ran.
5. Your Python, PyTorch and NumPy versions:

   ```bash
   python -c "import sys, torch, numpy; print(sys.version, torch.__version__, numpy.__version__)"
   ```

6. Anything unexpected:
   - a `Traceback` or warning;
   - a person/subject count other than 14 (Seeing Red) or 17 (Zenodo);
   - numbers far outside the ranges above.

Send these to the project lead directly. Don't commit result files or logs to the repository, and don't post them publicly.

## 10. Project rules

- Open-access data only.
- PPG is the only model input.
- Don't change defaults, settings or code without asking first.
- Report only what your own runs produced.
- Run one job at a time. About 12 threads is fastest on a CPU.
