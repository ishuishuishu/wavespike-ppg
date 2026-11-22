# Where this code comes from

Everything in `code/` is a copy of the project's own code, so this folder runs on its own. It is copied **unchanged**; don't edit it.

| File here | Copied from (repository) | Used for |
|---|---|---|
| `bio_run.py` | `biometrics/bio_run.py` | Training, evaluation, summaries, `--save-model` |
| `spiking_wavelets.py` | `biometrics/spiking_wavelets.py` | Imported by `bio_run.py` (it isn't used by the lean SNN itself) |
| `prepare_seeing_red.py` | `biometrics/prepare_seeing_red.py` | Seeing Red videos → PPG |
| `prepare_zenodo.py` | `biometrics/prepare_zenodo.py` | Zenodo wrist PPG → identity windows |
| `cells/02.py`, `03.py`, `11.py`, `13.py`, `15.py`, `17.py` | `notebook/cells/` | The model and training code; `bio_run.py` runs these cells |
| `load_model.py` | written for this handoff | Rebuilds a saved model and runs it |

- **Snapshot:** taken at repository commit `f5d5205`, the code behind the earlier runs plus the opt-in `--save-model` option.
- **How `bio_run.py` finds the cells:** it looks for `../notebook/cells` first and then `cells/` next to itself. Inside this folder, it uses `code/cells/`.
