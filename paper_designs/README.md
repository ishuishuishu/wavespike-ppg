# Spiking-wavelet papers P2–P5, rebuilt for PPG

This folder has everything needed to run our PPG versions of four published spiking-wavelet designs (P2, P3, P4, P5) and compare them with our own SNN. It is self-contained: copy just this folder to a server and follow this guide top to bottom.

## Important: what "recreate" means here

None of these four papers is about PPG:
- P2 is a theory paper tested on synthetic signals;
- P3 is EEG;
- P4 is image restoration;
- P5 is image and event-camera classification.

So we can't rerun their experiments. Instead, each paper's **wavelet design** is rebuilt for 1-D PPG and run inside **our** biometric pipeline: same data, folds, seeds, training recipe and metrics. Within a comparison only the wavelet part changes.

- Results from this folder are **our runs of designs rebuilt from the papers' text**. They are not the papers' own numbers, and must not be reported as such.
- Every choice a paper leaves open (a kernel size, a time constant, an initial value) is written in the docstring at the top of `code/spiking_wavelets.py`.

## The designs

| Paper | Design | Variant | How it's built for PPG |
|---|---|---|---|
| **P2**: Pedersen et al., *Scale-covariant spiking wavelets*, arXiv 2602.02020 (ICASSP 2026) | Cascaded leaky integrators (difference of time-causal kernels, DoT) | `x_p2_dot` | Replaces our learnable wavelet with a fixed cascade of K = 6 integrators (c = 2, τ_max = 0.04 s²), covering 0.8–25 Hz like our transform. Each band is the difference of adjacent stages, and the last stage is the low-pass band. Our spike encoder and recurrent SNN are unchanged |
| **P4**: Zhao et al., *Spiking pyramid wavelet transformation*, arXiv 2606.18644 (Pattern Recognition 2026) | Fixed Haar wavelet + small learnable separable convolution per sub-band | `x_p4_sep` | Fixed Haar transform, then a depthwise conv (kernel 5, our choice; the paper doesn't give one) on each sub-band and a pointwise conv across sub-bands, both starting as identity. Then our SNN |
| Baseline for P3 and P5 | Spikformer spiking self-attention | `x_stf_ssa` | A small 1-D spiking transformer: patches of 4 samples (125 tokens), width 32, 1 block, 4 time steps. P3 and P5 both modify this attention, so it's the reference they're compared with |
| **P3**: Yuan, Wei, Liu, *SNNs for EEG signal analysis using wavelet transform* (SpikeWavformer), Front. Neurosci. 2025 | Wavelet branch fused with spiking self-attention by concatenation (SWSA) | `x_stf_swsa` | The same transformer, with the attention output concatenated with a Haar → conv → inverse-Haar branch. Spike threshold 0.5, as the paper states |
| **P5**: Fang et al., *Spiking Wavelet Transformer* (SWformer), ECCV 2024 | Frequency-aware token mixer with ternary spikes (FATM) | `x_stf_fatm` | The same transformer, with attention replaced by: Haar → block-diagonal weights (2 blocks) → inverse Haar, using ternary {−1, 0, 1} spikes, plus a depthwise-conv branch and a pointwise branch, summed |

All five also feed two amplitude features (signal level and pulse size) into the output layer, like our own models (`amp_fusion`).

## Checklist

1. [ ] Set up Python and the data folder (§4.1).
2. [ ] Download and prepare Seeing Red; check it says **14 users** (§4.2).
3. [ ] Download and prepare Zenodo; check it says **17 subjects** (§4.3).
4. [ ] Run the smoke test; check there's no `Traceback` (§5).
5. [ ] Run all designs: `bash run_all.sh` (§6).
6. [ ] Print the results: `bash check_results.sh` (§7).
7. [ ] Check the saved models: `bash check_models.sh`, with 10 × `runs OK` (§8).
8. [ ] Send back the deliverables (§9).

Run every command **from inside this folder** (`cd paper_designs`). Run one job at a time.

## Files in this folder

| File | Use |
|---|---|
| `README.md` | This guide |
| `run_design.sh` | Runs one design on one dataset (6 runs) and saves the fold 0, seed 0 model |
| `run_all.sh` | Runs all 5 designs on both datasets, one after another (10 jobs, 60 runs) |
| `check_results.sh` | Prints the summary tables |
| `check_models.sh` | Reloads every saved model and runs it once |
| `code/load_model.py` | Rebuilds one saved model and runs it (used by `check_models.sh`; the starting point for the Pi) |
| `requirements.txt` | Python package versions used for our runs |
| `code/spiking_wavelets.py` | The P2–P5 designs, with every gap-filling choice documented |
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
- **Repeats.** 3 folds (0, 1, 2) × 2 training seeds (0, 1) = 6 runs per design per dataset.
- **Windows.** 10 s windows. Validation and test windows start every 2 s.
  - Seeing Red only: training windows start every 0.5 s (`--train-stride 0.5`).
  - Zenodo: training windows use the default 2 s.
- **Training recipe** (the same as our SNN): 150 epochs, 1,600 class-balanced windows per epoch, no early stopping, and the **last epoch** is reported. No augmentation.
- **Metrics:** top-1 accuracy, macro-F1, EER (equal error rate, lower is better) and FAR/FRR, for decisions on 1, 3 and 5 windows. We quote the **last epoch at 5 windows**.

## 2. Datasets

| Dataset | Used for | Licence |
|---|---|---|
| Seeing Red (Lovisotto et al., CVPRW 2020) | Across sittings, 14 people | ODC-PDDL (open) |
| Zenodo 23597128 (long-term wrist PPG) | Across days, 17 people | CC BY-NC-SA 4.0 (open, non-commercial) |

## 3. What has already been run

Nothing in this folder has been run yet: all 60 runs are new. For comparison, our own models on the same data (last epoch, 5 windows, mean of 6 runs unless noted):

| Our model | Zenodo EER | Seeing Red EER |
|---|---|---|
| Our SNN, fully learnable wavelet (`x_tau_amp`) | 15.6% | 28.7% (3 of 6 runs) |
| Our SNN, fixed db4 wavelet (`x_fixed_tau_amp`) | 16.1% | 24.7% |
| Our SNN, no wavelet (`x_raw_tau_amp`) | 20.2% | 29.5% |

The comparison that matters for P2 and P4 is against these SNN rows (same core, different wavelet). For P3 and P5 it's against `x_stf_ssa` (same transformer, without the wavelet part).

## 4. Setup (once)

### 4.1 Python and data folder

```bash
cd paper_designs
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
mkdir -p ~/wavespike_data/raw ~/wavespike_data/processed
export WAVESPIKE_DATA_ROOT=~/wavespike_data
```

- Python 3.13 or newer, CPU only. Our runs used Python 3.14 and PyTorch 2.15.
- If an exact version in `requirements.txt` isn't available, install the nearest one and note it in your report.
- **In every new shell**, `cd paper_designs`, then run `source .venv/bin/activate` and `export WAVESPIKE_DATA_ROOT=~/wavespike_data` again.
- If you already prepared the data for `lean_snn`, point `WAVESPIKE_DATA_ROOT` at the same folder and skip §4.2–4.3.

**Your data will match ours exactly.** Both preparation scripts are deterministic, so your folds are our folds. We checked this for Zenodo: a fresh preparation was identical to ours.

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

Each design trains for 2 epochs on fold 0 only:

```bash
for V in x_p2_dot x_p4_sep x_stf_ssa x_stf_swsa x_stf_fatm; do
  python -I code/bio_run.py --dataset zenodo_bio --variants $V --folds 0 --seeds 0 --smoke
done
BIO_CHANNEL=luma python -I code/bio_run.py --dataset seeing_red --variants x_p2_dot --folds 0 --seeds 0 --train-stride 0.5 --smoke
```

Each must print `identities 17` (or `identities 14` for Seeing Red), then one result line, with no `Traceback`.

## 6. Run the designs

**Everything, in order** (10 jobs: each design on Zenodo, then on Seeing Red):

```bash
bash run_all.sh            # optional thread count: bash run_all.sh 8
```

**Or one job at a time**, in any order:

```bash
bash run_design.sh x_p2_dot zenodo
bash run_design.sh x_p2_dot seeing_red
# ... and the same for x_p4_sep, x_stf_ssa, x_stf_swsa, x_stf_fatm
```

That is exactly, for example:

```bash
python -I code/bio_run.py --dataset zenodo_bio --variants x_p2_dot --folds 0,1,2 --seeds 0,1 --threads 12
BIO_CHANNEL=luma python -I code/bio_run.py --dataset seeing_red --train-stride 0.5 --variants x_p2_dot --folds 0,1,2 --seeds 0,1 --threads 12
```

**Time.** Not measured for these designs on a quiet machine. Similar designs took roughly 10–40 min per run; with 60 runs, plan for **one to two days** in total. The transformer designs (`x_stf_*`) may be slower than P2 and P4.

**Output:**
- One result file per run, `<variant>_f<fold>_ts<seed>.json`, in:
  - `$WAVESPIKE_DATA_ROOT/results/bio_zenodo_bio_w10/` (Zenodo);
  - `$WAVESPIKE_DATA_ROOT/results/bio_seeing_red_luma_w10_trs0.5/` (Seeing Red).
- One trained model per job, from **fold 0, seed 0 only**: `<results folder name>__<variant>_f0_ts0.pt` (about 30–50 KB), in `$WAVESPIKE_DATA_ROOT/models/paper_designs/`. That run prints `saved model ...` when the file is written; the other 5 runs of the job save no model.
- A log per job in `logs/`.

The run commands above are what `run_design.sh` runs, **plus** `--save-model $WAVESPIKE_DATA_ROOT/models/paper_designs --save-runs 0:0 --save-last-only`. If you type a command yourself, add those options so the model is saved.

**If interrupted:** run the same command again. Finished runs (those with a result file) are skipped; an interrupted run starts again from epoch 0. If a run has a result file but no model file, tell the project lead; don't delete the result file to force a rerun. Don't delete or edit result files.

## 7. Check the results

```bash
bash check_results.sh
```

- It prints one table per dataset, with a block of 3 rows (n = 1, 3, 5 windows) for each design.
- Read the rows where `n = 5`, the `EER% b/l` column, after the slash (last epoch). `top1 b/l` after the slash is the matching top-1 accuracy.
- There's no "correct" number for these designs. Report a mean EER outside about 12–40% as unexpected.

## 8. Saved models (for the Raspberry Pi later)

Only one model per design per dataset is saved, **10 in total**: the fold 0, seed 0 run, last-epoch weights. That's enough to run each design on the Pi. The other runs are needed only for the results, not for the Pi.

Each `.pt` file holds:
- `last_state`: the weights after the last epoch. **This is the model we report**, and the one for the Pi. (`best_state` is left empty.)
- `cfg`: every model setting, including which design it is.
- `identities`: the person IDs, in class order.
- `amp_norm`: the mean and standard deviation used to scale the two amplitude features (computed on that fold's training data).
- The variant, dataset, fold, seed and best epoch.

Saving changes nothing else: training, results and result files are exactly as without it.

Check every model rebuilds and runs:

```bash
bash check_models.sh
```

- It must list **10** models, each ending in `runs OK`.
- The predicted identity it prints is for a synthetic pulse and means nothing; only `runs OK` matters.
- To check one file: `python -I code/load_model.py $WAVESPIKE_DATA_ROOT/models/paper_designs/<file>.pt`.

**On the Pi (later, not part of this job):** the models run with PyTorch, which installs on a 64-bit Raspberry Pi OS. The project's NumPy-only Pi code covers only our own SNN, not these designs.

## 9. Deliverables (what to send back)

| # | Deliverable | Where | Count |
|---|---|---|---|
| 1 | Zenodo result files `x_{p2_dot,p4_sep,stf_ssa,stf_swsa,stf_fatm}_f{0,1,2}_ts{0,1}.json` | `$WAVESPIKE_DATA_ROOT/results/bio_zenodo_bio_w10/` | 30 |
| 2 | Seeing Red result files, same names | `$WAVESPIKE_DATA_ROOT/results/bio_seeing_red_luma_w10_trs0.5/` | 30 |
| 3 | Trained models `*__x_*_f0_ts0.pt` (one per design per dataset) | `$WAVESPIKE_DATA_ROOT/models/paper_designs/` | 10 |
| 4 | Output of `bash check_models.sh` (10 × `runs OK`) | your terminal | 1 |
| 5 | Logs | `logs/` | 10 |
| 6 | Output of `bash check_results.sh` | your terminal | 1 |
| 7 | Commit hash (or the name of the folder/zip you received), plus Python, PyTorch and NumPy versions | `git log --oneline -1`; `python -c "import sys, torch, numpy; print(sys.version, torch.__version__, numpy.__version__)"` | 1 |
| 8 | Notes on anything unexpected: tracebacks, wrong person counts, odd numbers, a model without `runs OK` | a short message | – |

**Done means:** 60 result files, 10 model files, 10 × `runs OK`, 10 logs and the summary. Send them to the project lead directly (a zip is fine; the models total under 1 MB). Don't commit them to the repository or post them publicly.

## 10. Project rules

- Open-access data only.
- PPG is the only model input.
- Don't change defaults, settings or code without asking first.
- Report only what your own runs produced, and label these as designs rebuilt from the papers' text, not the papers' results.
- Run one job at a time. About 12 threads is fastest on a CPU.
