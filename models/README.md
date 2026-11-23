# Trained models

| File | What it is | Runs with |
|---|---|---|
| `bio_zenodo_bio_w10__x_aug_tau_amp_f0_ts0.pt` | Our best SNN (fully learnable wavelet, augmentation, amplitude input), Zenodo across days, fold 0, seed 0, last-epoch weights | PyTorch: `python -I lean_snn/code/load_model.py <file>` |
| `wavespike_best_zenodo.npz` | The same model converted for the Raspberry Pi | NumPy only: `python3 edge/rpi_infer.py <file>` |
| `bio_seeing_red_luma_w10_trs0.5__x_aug_tau_amp_f0_ts0.pt` | Our best SNN, Seeing Red (phone camera) across sittings, fold 0, seed 0, last-epoch weights. Test EER 28.0 / 25.0 / 22.4% and top-1 33.8 / 31.9 / 33.3% on 1 / 3 / 5 windows; the original run of this split gave 27.5 / 25.0 / 24.8% EER | PyTorch (`load_model.py`) |
| `wavespike_best_seeing_red.npz` | The same model for the Raspberry Pi. Needs `level=` (mean camera brightness of the window, before high-pass) in `classify_window`. Check on 200 test windows: same identity on 100%; 99% of windows equal to PyTorch within 0.1%, 2 differ by up to 0.6 from a spike flipping at its threshold | NumPy only (`rpi_infer.py`) |
| `bio_zenodo_bio_w10__x_p1_doe_learn_bestval.pt` | P1 DoE with learnable scales (our extension), Zenodo; best of 6 runs by validation EER (fold 2, seed 1, val EER 13.7%) | PyTorch (`load_model.py`) |
| `bio_seeing_red_luma_w10_trs0.5__x_p1_doe_learn_bestval.pt` | Same, Seeing Red (fold 2, seed 0, val EER 16.1%) | PyTorch |
| `bio_zenodo_bio_w10__x_p1_doe_penc_bestval.pt` | P1 DoE with the paper's own encoder, Zenodo (fold 2, seed 1, val EER 11.8%) | PyTorch |
| `bio_seeing_red_luma_w10_trs0.5__x_p1_doe_penc_bestval.pt` | Same, Seeing Red (fold 0, seed 1, val EER 15.6%) | PyTorch |

P1 models are designs rebuilt from the paper's text (Pedersen et al.), not the paper's own models. "Best" is chosen on validation data only; the reported P1 results are the mean of all 6 runs. Their 6-run test means (last epoch, 5 windows): learnable scales 16.4% (Zenodo) / 27.4% (Seeing Red) EER; paper encoder 17.0% / 23.8%. The P1 front ends are not exportable to the NumPy Pi format; they run with PyTorch.

**Test results of the best-SNN run** (fold 0 seed 0, last epoch; our own run): EER 23.4 / 19.9 / 18.4% and top-1 26.9 / 37.1 / 36.3% for decisions on 1 / 3 / 5 windows. Fold 0 is the hardest Zenodo split; the published 6-run mean (16.0% EER at 5 windows) is unchanged. The original fold-0 seed-0 run gave 23.6 / 19.9 / 18.1% EER, so this retrained copy matches it within normal run-to-run variation.

**Pi conversion check** (`edge/export_pi.py`, 200 test windows of this run): NumPy and PyTorch give the same identity on 100% of windows; the largest score difference is 0.02 on a score range of about 20, from occasional single-spike differences caused by float rounding.

**Input:** one 10 s window at 50 Hz, prepared like the training data (`biometrics/prepare_zenodo.py`). The amplitude input is computed from the raw window inside `rpi_infer.py`. The 17 identities are Zenodo subject IDs, stored in the file.

Zenodo data licence: "Other (Non-Commercial)"; these weights are trained on it, so use them for research only.
