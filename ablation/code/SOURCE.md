# Where this code comes from

Everything in `code/` is a copy of the project's own code, so this folder runs on its own. It is copied **unchanged**; don't edit it.

| File here | Copied from (repository) | Used for |
|---|---|---|
| `bio_run.py` | `biometrics/bio_run.py` | Training, evaluation and summaries; defines all eight ablation variants |
| `spiking_wavelets.py` | `biometrics/spiking_wavelets.py` | Imported by `bio_run.py`; not used by the ablation models |
| `prepare_seeing_red.py` | `biometrics/prepare_seeing_red.py` | Seeing Red videos → PPG |
| `prepare_zenodo.py` | `biometrics/prepare_zenodo.py` | Zenodo wrist PPG → identity windows |
| `cells/02.py`, `03.py`, `11.py`, `13.py`, `15.py`, `17.py` | `notebook/cells/` | Our SNN (including the learnable wavelet), training loop and recipe; `bio_run.py` runs these cells |

- **Snapshot:** taken at repository commit `29d86a5`.
- **How `bio_run.py` finds the cells:** it looks for `../notebook/cells` first and then `cells/` next to itself. Inside this folder, it uses `code/cells/`.
