# Archive: other scripts used during the project

These scripts were run during development. None are needed to run the main code. They are kept so that every reported number can be traced back to the script that produced it. The launch scripts for the compute machine are not included.

| Script | What it did |
| --- | --- |
| `analysis/bio_fuse.py` | Offline score fusion and ensembling of saved biometric results, with the weight picked on validation; `ZNORM=1` adds per-identity z-norm. Produced the SVM + GRU fusion result (17.8% EER on Seeing Red) and the z-norm check. Usage: `python -I archive/analysis/bio_fuse.py <results_dir> <classical or -> <neural1+neural2> [ckpt] [seeds]` |
| `analysis/allmetrics.py` | Collects every metric from every result file into one JSON, the data behind `docs/results.html`. Usage: `python -I archive/analysis/allmetrics.py <data_root> allmetrics.json` |
| `analysis/summ.py` | Compact AF summary from the notebook's result files. **Known bug:** it reads every `<dataset>_v3_*` results folder together, so it mixed a 7-subject pilot into the Zenodo AF numbers. Use `allmetrics.py` instead |
| `analysis/oof_partial.py` | Early AF summary of partly finished runs (MIMIC PERform) |

The notebook builder is `notebook/build.py` (`python3 notebook/build.py` rebuilds the notebook from `notebook/cells/`).
