# WaveSpike-PPG

A small spiking neural network (SNN) with a learnable wavelet front end for photoplethysmography (PPG), designed to run on edge devices such as a Raspberry Pi. It was first built for atrial fibrillation (AF) detection and is now evaluated on PPG biometrics (recognising a person from their pulse).

**Start here:** [`docs/HANDOFF.md`](docs/HANDOFF.md) explains the project, every result and the open tasks. [`docs/results.html`](docs/results.html) has every metric from every run; download it and open it in a browser. [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) is the running record of every experiment and the plan for the next ones.

## What's in the repository

| Folder | Contents |
| --- | --- |
| `notebook/` | `wavespike_ppg_colab_v3.ipynb`: the model plus the AF pipeline (data, training, evaluation, NumPy export). `cells/` holds the same code as separate files, which the biometric scripts import |
| `biometrics/` | `bio_run.py` (biometric experiments: datasets, splits, baselines, metrics), `paper_repro.py` (published methods under their own protocols), `paper_models.py` (published CWT networks built from their papers), `prepare_seeing_red.py` and `prepare_zenodo.py` (dataset preparation) |
| `edge/` | `rpi_infer.py` (NumPy-only inference) and `pi_bench.py` (Raspberry Pi latency, memory and energy benchmark), `energy_estimate.py` (hardware-free energy estimates from operation counts) |
| `docs/` | Team handoff and the full results page |
| `archive/` | Other analysis scripts used during development; see `archive/README.md` |
| `manuscript/` | Manuscript drafts (Word) |

## Quick start

1. Install Python 3.13 or newer with CPU PyTorch, NumPy, SciPy, scikit-learn, pandas, h5py, PyWavelets and opencv-python-headless.
2. Download the datasets you need into a data folder (see below) and point `WAVESPIKE_DATA_ROOT` at it.
3. Run an experiment, for example:

```bash
export WAVESPIKE_DATA_ROOT=$PWD/wavespike_data
python -I biometrics/prepare_seeing_red.py <seeing_red_videos_dir> $WAVESPIKE_DATA_ROOT/raw/seeing_red/sr_rgb50.npz
BIO_CHANNEL=luma python -I biometrics/bio_run.py --dataset seeing_red --variants beat_svm_amp,x_aug_tau_amp --train-stride 0.5 --threads 12
BIO_CHANNEL=luma python -I biometrics/bio_run.py --dataset seeing_red --train-stride 0.5 --summary --variants beat_svm_amp,x_aug_tau_amp
```

Results are written to `$WAVESPIKE_DATA_ROOT/results/` as one JSON file per run. The AF pipeline runs from the notebook. On CPU, one process with about 12 threads is fastest.

## Data (not included)

| Dataset | Used for | Licence |
| --- | --- | --- |
| Seeing Red (Lovisotto et al., CVPRW 2020), Oxford Research Archive | Biometrics across sittings | ODC-PDDL |
| Zenodo 23597128 (long-term wrist PPG with ECG AF labels) | AF detection; biometrics across days | CC BY-NC-SA 4.0 |
| MIMIC PERform AF (Zenodo 6807403) | AF detection; same-session biometrics | Check the record |
| BIDMC (PhysioNet) | Same-session biometrics on clean data | ODC-BY 1.0 |
| PhysioNet/CinC Challenge 2015 | Sanity check only | Open |

Expected layout under `WAVESPIKE_DATA_ROOT`:

- `raw/seeing_red/sr_rgb50.npz`
- `raw/zenodo_23597128/`
- `raw/bidmc/bidmc.zip`
- `processed/` (caches built by the scripts and the notebook)
