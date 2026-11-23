# Where this code comes from

Everything in `code/` is a copy of the project's own code, so this folder runs on its own. It is copied **unchanged**; don't edit it.

| File here | Copied from (repository) | Used for |
|---|---|---|
| `bio_run.py` | `biometrics/bio_run.py` | Training, evaluation, summaries; defines the P6 variants `x_p6_haar`, `x_p6_db2`, `x_p6_db8` (and `x_p6_sym4`, `x_p6_coif1`) |
| `spiking_wavelets.py` | `biometrics/spiking_wavelets.py` | **The P6 design itself:** `class FixedSWT`. Its docstring lists every choice the paper leaves open |
| `prepare_seeing_red.py` | `biometrics/prepare_seeing_red.py` | Seeing Red videos → PPG |
| `prepare_zenodo.py` | `biometrics/prepare_zenodo.py` | Zenodo wrist PPG → identity windows |
| `cells/02.py`, `03.py`, `11.py`, `13.py`, `15.py`, `17.py` | `notebook/cells/` | Our SNN, training loop and recipe; `bio_run.py` runs these cells |
| `load_model.py` | `paper_designs/code/load_model.py` | Rebuilds a saved model and runs it |

- **Snapshot:** taken at repository commit `55556fb`.
- **How `bio_run.py` finds the cells:** it looks for `../notebook/cells` first and then `cells/` next to itself. Inside this folder, it uses `code/cells/`.
