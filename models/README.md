# Trained models

| File | What it is | Runs with |
|---|---|---|
| `bio_zenodo_bio_w10__x_aug_tau_amp_f0_ts0.pt` | Our best SNN (fully learnable wavelet, augmentation, amplitude input), Zenodo across days, fold 0, seed 0, last-epoch weights | PyTorch: `python -I lean_snn/code/load_model.py <file>` |
| `wavespike_best_zenodo.npz` | The same model converted for the Raspberry Pi | NumPy only: `python3 edge/rpi_infer.py <file>` |

**Test results of this run** (fold 0 seed 0, last epoch; our own run): EER 23.4 / 19.9 / 18.4% and top-1 26.9 / 37.1 / 36.3% for decisions on 1 / 3 / 5 windows. Fold 0 is the hardest Zenodo split; the published 6-run mean (16.0% EER at 5 windows) is unchanged. The original fold-0 seed-0 run gave 23.6 / 19.9 / 18.1% EER, so this retrained copy matches it within normal run-to-run variation.

**Pi conversion check** (`edge/export_pi.py`, 200 test windows of this run): NumPy and PyTorch give the same identity on 100% of windows; the largest score difference is 0.02 on a score range of about 20, from occasional single-spike differences caused by float rounding.

**Input:** one 10 s window at 50 Hz, prepared like the training data (`biometrics/prepare_zenodo.py`). The amplitude input is computed from the raw window inside `rpi_infer.py`. The 17 identities are Zenodo subject IDs, stored in the file.

Zenodo data licence: "Other (Non-Commercial)"; these weights are trained on it, so use them for research only.
