# Wavelet ablation: handoff

This folder is the complete handoff for the **wavelet ablation** of our spiking neural network (SNN) for PPG biometrics. It contains:
- everything we have trained and measured so far;
- what's still missing, and why it matters;
- all the code;
- step-by-step instructions to finish the missing runs.

It is self-contained: copy just this folder to a server and follow this guide top to bottom.

## Your job, in short

1. Read §1–§4 to understand what the ablation is and where it stands (about 15 minutes).
2. Prepare two open datasets, Seeing Red and Zenodo (§5).
3. Run the **21 missing training runs** with one command, `bash run_missing.sh`, roughly 5–14 hours on a quiet machine (§6).
4. Print the completed table with `python -I ablation_table.py` (§7).
5. Send back the deliverables below (§8).

## Deliverables (what to send back)

| # | Deliverable | Where | Count |
|---|---|---|---|
| 1 | Seeing Red result files `x_tau_amp_f{0,1,2}_ts1.json` | `$WAVESPIKE_DATA_ROOT/results/bio_seeing_red_luma_w10_trs0.5/` | 3 |
| 2 | Seeing Red result files `x_shared_tau_amp_f{0,1,2}_ts{0,1}.json` | same folder | 6 |
| 3 | Zenodo result files `x_aug_raw_tau_amp_f{0,1,2}_ts{0,1}.json` | `$WAVESPIKE_DATA_ROOT/results/bio_zenodo_bio_w10/` | 6 |
| 4 | Zenodo result files `x_aug_fixed_tau_amp_f{0,1,2}_ts{0,1}.json` | same folder | 6 |
| 5 | Logs, one per job | `logs/` | 4 |
| 6 | Printout of `python -I ablation_table.py`: the full table with no "missing" cells | `ablation_table.txt` (§8) | 1 |
| 7 | Commit hash (or the name of the zip you received), plus Python, PyTorch and NumPy versions | `versions.txt` (§8) | 1 |
| 8 | A short note on anything unexpected, or "nothing unexpected" | a message | 1 |

**Done means:** 21 result files, 4 logs, a table where every cell shows 6/6 runs, your versions and the note. §8 has one command that zips it all.

No trained models are needed for this job.

---

## 1. What the ablation is and why it matters

Our model is a small SNN (about 6k parameters) that identifies a person from 10 s of PPG. Before the spiking part, a **wavelet front end** splits the signal into frequency bands. The paper's central claim depends on that front end: *does a learnable wavelet help, and how much of the model's accuracy comes from it?*

The ablation answers this by changing **only the front end** and keeping everything else identical: data, folds, seeds, spiking core, training recipe, metrics. There are four front ends, each trained with and without augmentation, on two datasets: 4 × 2 × 2 = 16 cells, 6 runs each.

### The four front ends

| Row | Variant (no aug / with aug) | What the front end does | Setting that changes |
|---|---|---|---|
| **No wavelet** | `x_raw_tau_amp` / `x_aug_raw_tau_amp` | The raw PPG goes straight into the spike encoder. No bands, no denoising threshold | `use_wavelet: False`, `use_ht: False` |
| **Fixed db4** | `x_fixed_tau_amp` / `x_aug_fixed_tau_amp` | A standard Daubechies-4 wavelet (8 taps) splits the signal into 6 bands (d1–d5 plus the approximation); the filter never changes | `learn_wavelet: False` |
| **Partly learnable** | `x_shared_tau_amp` / `x_aug_shared_tau_amp` | Starts as db4, but **one** filter, shared by all 5 levels, is learned during training | `per_level: False` |
| **Fully learnable** (our main model) | `x_tau_amp` / `x_aug_tau_amp` | Starts as db4, and **each level** learns its own filter | the defaults |

What all eight variants share:
- the learnable denoising threshold (except "no wavelet", which has no bands to threshold);
- the ± spike encoder;
- a recurrent spiking layer with per-neuron time constants (`tau_het`);
- two amplitude features fused into the output (`amp_fusion`);
- the same training recipe (§2).

"With augmentation" (`augment: True`) adds random amplitude and time changes to the training windows only.

### What each comparison tells us

- **No wavelet vs any wavelet:** does splitting into bands help at all?
- **Fixed vs partly vs fully learnable:** does learning the filter help, and is one shared filter enough?
- **With vs without augmentation:** does the answer hold whether or not the training data is augmented? This is why both columns must be complete. An earlier version of our results page compared rows with and without augmentation, mixing two effects; this table fixes that.

## 2. How the tests work (same for every run)

- **Task.** Closed-set identification: each person is one class.
- **Split by session.** Training, validation and test windows come from **different sessions**, so the model has to recognise people on a day or sitting it hasn't seen:
  - **Zenodo** (17 people, wrist PPG): a session is a recording day.
  - **Seeing Red** (14 people, fingertip on a phone camera): a session is a sitting.
- **Repeats.** 3 folds (0, 1, 2) × 2 training seeds (0, 1) = 6 runs per cell.
  - The folds rotate which sessions are held out.
  - The seed changes only the starting weights and the sampling.
- **Windows.** 10 s at 50 Hz, z-scored per window. Validation and test windows start every 2 s.
  - Seeing Red only: training windows start every 0.5 s (`--train-stride 0.5`).
  - Zenodo: training windows use the default 2 s.
- **Training.** 150 epochs, 1,600 class-balanced windows per epoch, no early stopping. The **last epoch** is reported.
- **Metrics.** Each is computed for decisions on 1, 3 and 5 consecutive windows:
  - EER (equal error rate; lower is better);
  - top-1 accuracy;
  - macro-F1;
  - FAR/FRR.

  **The headline number is the EER at 5 windows, last epoch**, averaged over the 6 runs.
- **Claim rule.** We say one front end is better than another only if it wins in **every one of the 6 paired repeats** (same fold and seed). A better average alone isn't enough.

## 3. Results so far (our runs)

These are all our own runs (75 so far), from the file `ablation_runs_so_far.csv` in this folder. Each cell shows the mean over runs, the number of runs out of 6, and the min–max range. **Missing** cells are your job.

### EER % at 5 windows, last epoch (lower is better)

| Front end | Zenodo, no aug | Zenodo, aug | Seeing Red, no aug | Seeing Red, aug |
|---|---|---|---|---|
| No wavelet | 20.2 (6/6, 18.6–21.6) | **missing** | 29.5 (6/6, 22.2–35.4) | 24.9 (6/6, 17.8–30.4) |
| Fixed db4 | 16.1 (6/6, 14.4–19.1) | **missing** | 24.7 (6/6, 15.7–34.8) | 23.8 (6/6, 15.6–30.4) |
| Partly learnable | 16.6 (6/6, 14.2–20.3) | 16.1 (6/6, 14.3–18.5) | **missing** | 23.9 (6/6, 17.8–30.4) |
| Fully learnable | 15.6 (6/6, 14.0–18.1) | 16.0 (6/6, 14.5–18.1) | 28.7 (**3/6**, 20.0–39.1) | 24.3 (6/6, 17.8–32.3) |

### Top-1 accuracy % at 5 windows, last epoch (higher is better)

| Front end | Zenodo, no aug | Zenodo, aug | Seeing Red, no aug | Seeing Red, aug |
|---|---|---|---|---|
| No wavelet | 30.2 | **missing** | 37.5 | 40.4 |
| Fixed db4 | 48.9 | **missing** | 41.5 | 38.9 |
| Partly learnable | 50.0 | 44.2 | **missing** | 34.9 |
| Fully learnable | 49.2 | 43.9 | 41.9 (3/6) | 37.5 |

### Paired comparison: fully learnable vs each other front end

Per repeat (same fold and seed), with the same augmentation setting. "Wins" means the fully learnable wavelet had the lower EER.

| Dataset | Augmentation | Compared with | Fully learnable wins | Mean EER difference | Verdict |
|---|---|---|---|---|---|
| Zenodo | no | No wavelet | **6 of 6** | −4.6 points | **Better in every repeat** |
| Zenodo | no | Fixed db4 | 5 of 6 | −0.5 | No consistent difference |
| Zenodo | no | Partly learnable | 3 of 6 | −1.0 | No consistent difference |
| Zenodo | aug | Partly learnable | 4 of 6 | −0.1 | No consistent difference |
| Zenodo | aug | No wavelet, fixed db4 | – | – | **Missing runs** |
| Seeing Red | no | No wavelet | 2 of 3 | −0.2 | Too few repeats |
| Seeing Red | no | Fixed db4 | 0 of 3 | +4.2 | Too few repeats |
| Seeing Red | no | Partly learnable | – | – | **Missing runs** |
| Seeing Red | aug | No wavelet | 2 of 6 | −0.6 | No consistent difference |
| Seeing Red | aug | Fixed db4 | 2 of 6 | +0.5 | No consistent difference |
| Seeing Red | aug | Partly learnable | 2 of 6 | +0.4 | No consistent difference |

### What this says so far (cautiously)

- **A wavelet clearly helps on Zenodo.** Without augmentation, every wavelet front end beats no wavelet on average by 3.6–4.6 EER points, and the fully learnable one wins in all 6 repeats. This is the one consistent result so far.
- **Learning the filter isn't yet shown to help.** Fixed, partly learnable and fully learnable are within about 1 point of each other, with no consistent winner. The fully learnable wavelet is not "better" by our rule anywhere except against no wavelet.
- **Seeing Red is noisy.** Ranges span 15–39% EER across repeats, so differences of 1–4 points can't be trusted. The no-augmentation column also has the most missing runs.
- **Augmentation** helps on Seeing Red: every row with both columns improves, by 0.9 to 4.6 points. On Zenodo it's mixed: fully learnable goes from 15.6 to 16.0, partly learnable from 16.6 to 16.1, and the other two rows are missing. That's why the Zenodo augmented column must be complete before drawing conclusions.

These are interim readings. Don't quote them as conclusions until the table is complete.

## 4. What's missing and where it stands

### The 21 missing runs

| Job | Dataset | Variant | Runs | Why it's needed |
|---|---|---|---|---|
| 1 | Seeing Red | `x_tau_amp` (fully learnable, no aug), **seed 1 only** | 3 | Our main model has only 3 of 6 runs there; seed 0 is done |
| 2 | Seeing Red | `x_shared_tau_amp` (partly learnable, no aug) | 6 | Empty cell; needed for the fixed vs partly vs fully comparison without augmentation |
| 3 | Zenodo | `x_aug_raw_tau_amp` (no wavelet, aug) | 6 | Empty cell; tests whether the wavelet still helps once augmentation is on |
| 4 | Zenodo | `x_aug_fixed_tau_amp` (fixed db4, aug) | 6 | Empty cell; fixed vs learnable with augmentation |

`run_missing.sh` runs these four jobs in this order. Any job can be run on its own (§6).

### Other gaps in this part of the project (not part of your job)

These are listed so the picture is complete. Don't start them unless the project lead asks.
- **Statistics.** So far we only count wins per repeat. The final write-up will add confidence intervals and a paired test across the 6 repeats, computed by a summary script that isn't written yet.
- **Wider fixed-wavelet row.** "Fixed db4" is one wavelet. A separate P6 job sweeps other fixed families (haar, db2, db8, sym4, coif1), which will show whether db4 was a lucky choice. sym4 (16.5%) and coif1 are done on Zenodo; haar, db2 and db8 are still running.
- **Energy per front end.** Energy estimates exist for the fully learnable and lean SNNs, but not yet for every ablation row.
- **Only two datasets.** The ablation covers Zenodo and Seeing Red (across sessions). It hasn't been run on the same-session datasets (BIDMC, MIMIC).
- **Epoch choice.** We report the last epoch. The best-epoch numbers are in the result files and the CSV but aren't used for claims.

## 5. Setup (once)

### 5.1 Python and data folder

```bash
cd ablation
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
mkdir -p ~/wavespike_data/raw ~/wavespike_data/processed
export WAVESPIKE_DATA_ROOT=~/wavespike_data
```

- Python 3.13 or newer, CPU only. Our runs used Python 3.14 and PyTorch 2.15.
- If an exact version in `requirements.txt` isn't available, install the nearest one and note it in your report.
- **In every new shell:** `cd ablation`, then `source .venv/bin/activate` and `export WAVESPIKE_DATA_ROOT=~/wavespike_data`.
- If you already prepared the data for `lean_snn` or `paper_designs`, point `WAVESPIKE_DATA_ROOT` at that same folder and skip §5.2–5.3.

**Your data will match ours exactly.** Both preparation scripts are deterministic, so your folds are our folds and your runs pair up with ours. We checked this for Zenodo: a fresh preparation was identical to ours.

### 5.2 Seeing Red (ODC-PDDL, open)

1. On the Oxford Research Archive (ora.ox.ac.uk), find the dataset for Lovisotto, Turner, Eberz and Martinovic, *Seeing Red: PPG Biometrics Using Smartphone Cameras* (CVPR Workshops 2020). Download the videos zip (about 590 MB) and unzip it. You need one subfolder per person with `.mp4` files.
2. Convert it:

   ```bash
   mkdir -p $WAVESPIKE_DATA_ROOT/raw/seeing_red
   python -I code/prepare_seeing_red.py <videos_folder> $WAVESPIKE_DATA_ROOT/raw/seeing_red/sr_rgb50.npz 12
   ```

   It must print **14 users**.

### 5.3 Zenodo (CC BY-NC-SA 4.0, open)

1. From <https://zenodo.org/records/23597128>, download every `*_PPG.mat` file into `$WAVESPIKE_DATA_ROOT/raw/zenodo_23597128/`. The `_ECG.mat` files aren't needed.
2. Build the windows:

   ```bash
   python -I code/prepare_zenodo.py $WAVESPIKE_DATA_ROOT
   ```

   It must end with `... 6000 windows, 17 subjects`.

### 5.4 Smoke test (a few minutes; writes no result files)

```bash
BIO_CHANNEL=luma python -I code/bio_run.py --dataset seeing_red --train-stride 0.5 --variants x_tau_amp,x_shared_tau_amp --folds 0 --seeds 1 --smoke
python -I code/bio_run.py --dataset zenodo_bio --variants x_aug_raw_tau_amp,x_aug_fixed_tau_amp --folds 0 --seeds 0 --smoke
```

Each must print `identities 14` (Seeing Red) or `identities 17` (Zenodo), then one line per variant, with no `Traceback`.

## 6. Run the missing jobs

**All four, in order:**

```bash
bash run_missing.sh            # optional thread count: bash run_missing.sh 8
```

**What it runs**, so you can also run any job on its own:

```bash
# Job 1: Seeing Red, fully learnable, no aug, seed 1 (3 runs)
BIO_CHANNEL=luma python -I code/bio_run.py --dataset seeing_red --train-stride 0.5 --variants x_tau_amp --folds 0,1,2 --seeds 1 --threads 12
# Job 2: Seeing Red, partly learnable, no aug (6 runs)
BIO_CHANNEL=luma python -I code/bio_run.py --dataset seeing_red --train-stride 0.5 --variants x_shared_tau_amp --folds 0,1,2 --seeds 0,1 --threads 12
# Job 3: Zenodo, no wavelet, aug (6 runs)
python -I code/bio_run.py --dataset zenodo_bio --variants x_aug_raw_tau_amp --folds 0,1,2 --seeds 0,1 --threads 12
# Job 4: Zenodo, fixed db4, aug (6 runs)
python -I code/bio_run.py --dataset zenodo_bio --variants x_aug_fixed_tau_amp --folds 0,1,2 --seeds 0,1 --threads 12
```

- Job 1 uses `--seeds 1` only: seed 0 already exists on our side. Don't rerun it.
- On Zenodo, **don't** pass `--train-stride`.

**Time.** On a quiet machine, expect roughly 15–40 minutes per run with 12 threads (our least-loaded run of this kind took 13 minutes). That's about **5–14 hours** for all 21 runs. Each run prints one result line when it finishes.

**Output:**
- one result file per run, `<variant>_f<fold>_ts<seed>.json`, in `$WAVESPIKE_DATA_ROOT/results/bio_seeing_red_luma_w10_trs0.5/` or `$WAVESPIKE_DATA_ROOT/results/bio_zenodo_bio_w10/`;
- one log per job in `logs/`.

**If interrupted:** run the same command again. Finished runs (those with a result file) are skipped; an interrupted run restarts from epoch 0. Don't delete or edit result files.

## 7. Check the results: the full table

```bash
python -I ablation_table.py
```

- **What it does:** merges our 75 runs (`ablation_runs_so_far.csv`) with your new result files (from `$WAVESPIKE_DATA_ROOT`), then prints:
  - the EER table;
  - the top-1 table;
  - the paired comparisons with their verdicts, as in §3.
- **When you're done:** every cell should show **6/6**, with no `missing`, and the Seeing Red "fully learnable, no aug" cell should go from 3/6 to 6/6.
- **Other views** (optional): `--n 1` or `--n 3` for decisions on 1 or 3 windows, `--checkpoint best` for best-epoch numbers.

There's no "correct" number for the missing cells. As a sanity check, Zenodo EERs have been 15–21% and Seeing Red 23–30% so far. Report a cell far outside that as unexpected.

## 8. Package and send the deliverables

These commands collect everything into `ablation_results.zip`:

```bash
cd ablation
python -I ablation_table.py > ablation_table.txt 2>&1
{ git log --oneline -1 2>/dev/null || echo "no git: <name of the zip you received>"
  python -c "import sys, torch, numpy; print(sys.version, torch.__version__, numpy.__version__)"; } > versions.txt
R=$WAVESPIKE_DATA_ROOT/results
zip -r ablation_results.zip ablation_table.txt versions.txt logs \
    $R/bio_seeing_red_luma_w10_trs0.5/x_tau_amp_f*_ts1.json \
    $R/bio_seeing_red_luma_w10_trs0.5/x_shared_tau_amp_f*_ts*.json \
    $R/bio_zenodo_bio_w10/x_aug_raw_tau_amp_f*_ts*.json \
    $R/bio_zenodo_bio_w10/x_aug_fixed_tau_amp_f*_ts*.json
```

Before sending, check:
- `ablation_table.txt` has no `missing` and every cell shows 6/6;
- `unzip -l ablation_results.zip` lists 21 `.json` files and 4 logs.

If `zip` isn't installed, use `sudo apt install zip` or any archive tool with the same files. Send the zip and your note to the project lead directly; don't commit it or post it publicly.

## 9. Files in this folder

| File | Use |
|---|---|
| `README.md` | This guide |
| `run_missing.sh` | Runs the 4 missing jobs (21 runs), one after another |
| `ablation_table.py` | Prints the full ablation table (our runs plus yours), with paired comparisons |
| `ablation_runs_so_far.csv` | Metrics of our 75 finished runs: every variant, fold, seed, epoch choice (last/best) and window count (1/3/5), plus parameters and run time |
| `requirements.txt` | Python package versions used for our runs |
| `code/bio_run.py` | Training, evaluation and summaries; defines all eight variants |
| `code/cells/` | Our SNN (including the learnable wavelet), training loop and recipe |
| `code/prepare_seeing_red.py`, `code/prepare_zenodo.py` | Turn the downloaded data into PPG windows |
| `code/spiking_wavelets.py` | Imported by `bio_run.py`; not used by these models |
| `code/SOURCE.md` | Where each code file was copied from |

`ablation_runs_so_far.csv` has one row per run × epoch choice × window count. Columns:
- `dataset`, `results_folder`, `variant`, `fold`, `seed`;
- `checkpoint` (`last` or `best`), `n_windows`;
- `top1`, `macro_f1`, `eer`, `eer_val` (validation EER), `far`, `frr`;
- `params`, `epochs_run`, `minutes` (run time on our machine, often under heavy load).

## 10. Project rules

- Open-access data only.
- PPG is the only model input.
- **Don't change any setting, default or line of code.** If something looks wrong, stop and ask.
- Report only what your own runs produced.
- Run one job at a time. About 12 threads is fastest on a CPU.
