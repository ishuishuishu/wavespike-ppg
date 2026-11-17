# WaveSpike-PPG: team handoff

## Summary

WaveSpike-PPG is a small spiking neural network (SNN) with a learnable wavelet front end, built to run on edge devices such as a Raspberry Pi. The input is PPG only. It was built for atrial fibrillation (AF) detection. That failed, so the project switched to PPG biometrics (recognising a person from their pulse).

**Where it stands:**

- **Main accuracy experiments:** finished.
- **Still running:** checks on clean hospital datasets and re-runs of published methods under their own protocols. Their status is marked in each table.
- **Raspberry Pi benchmark** (`edge/pi_bench.py`): ready. It measures metrics 6–10 of the team's METRICS_REPORT.md (latency, measured energy, peak RAM, storage, SynOps). It needs Pi access and a power meter (INA226 or a USB meter) on the Pi's supply.

**Headline results** (all from our own runs):

- **Biometrics across days (Zenodo wrist, 17 people):**
  - Our SNN reaches 15.6% EER from 5 windows (about 50 s of pulse).
  - It beats every classical method, a same-size GRU and the SNN without a wavelet, in 6 of 6 repeats.
  - A fixed CWT scalogram + CNN is better at 14.0%. It is about 4× bigger and needs roughly 5× more compute (estimate).
- **Biometrics on Seeing Red (phone camera, 14 people, across sittings):** the beat-feature SVM is best at 19.6% EER; our best SNN gets 21.8%.
- **Clean hospital PPG, same session (BIDMC, MIMIC):** the classical recreations reach 94–96% top-1 accuracy from one 10 s window and 1.0–1.8% EER, in line with published 90%+ figures.
- **Wavelet:** a wavelet front end cuts error by about 4–6 points. Making it learnable, partly or fully, adds no accuracy.
- **AF detection:**
  - MIMIC PERform AF: a 3-feature heartbeat-interval baseline scores 0.919 AUROC, against 0.668 for our SNN.
  - Wrist AF data: every method is at chance.

**Proposed paper framing:** accuracy per unit of energy.

- On the hard across-days test, our SNN comes close to a 4× larger CNN.
- It beats classical wavelet pipelines and an equal-size GRU.
- It streams sample by sample.
- The Pi benchmark has to turn the compute estimate into measured energy and latency.

The full results, with every metric, are in [`docs/results.html`](results.html) (open it in a browser).

## Rules the team follows

1. **Open-access data only.** Credentialed data (MIMIC-III-Ext-PPG, DeepBeat, UConn, Pulsewatch, Biosec1) is not used.
2. **Don't change defaults silently.** New settings are opt-in and listed. The notebook defaults are untouched; all biometric changes are options in `biometrics/bio_run.py`.
3. **The all-repeats rule.** An improvement is claimed only if the paired difference has the same sign in every repeat. That means 6 repeats: 3 fold seeds × 2 training seeds for AF, and 3 folds × 2 seeds for biometrics.
4. **No tuning on test.** Thresholds are set on validation data (2% FAR for biometrics), and model selection uses validation EER. Anything picked by looking at test numbers is labelled optimistic.
5. **Input is PPG only.** ECG is used only as the source of AF labels in the Zenodo data.

## Progress

1. **Setup.** The v3 notebook was run end to end. The determinism check passed.
2. **Speed.** One process with 12 CPU threads was fastest; extra parallel processes were slower.
3. **AF evaluation.** 300 runs (10 variants × 6 repeats × 5 folds) took about 5 h. Manuscript v0.1–v0.3 followed. The CinC 2015 sanity task was run.
4. **Edge tooling.** The Pi benchmark was built from METRICS_REPORT.md, and a spike counter was added to the NumPy export.
5. **More AF attempts.**
    - 8 new variants were screened on MIMIC; the best, `tau_aug_slow`, reached 0.81 AUROC on one optimistic dev repeat.
    - Zenodo wrist AF was downloaded (20 subjects, 19 usable).
    - 15 variants were run on it; all were at chance.
6. **Switch to biometrics.** Seeing Red was converted to PPG. A red-channel saturation bug was found and fixed by switching to luma.
7. **Accuracy round on Seeing Red** (3 stages). Neural EER fell from about 31–34% to 22–24%.
8. **Across days on Zenodo.** With the improved recipe the SNN reached 15.6% EER, beating the SVM, the GRU and the no-wavelet SNN in 6/6 repeats.
9. **Published-style baselines:** DWT + kNN/SVM/LDA, CWT scalogram + CNN, and the partly learnable wavelet. The CWT-CNN beats our SNN.
10. **Wavelet-packet front end** (16 bands): no gain.
11. **Same-session protocol and clean datasets (BIDMC, MIMIC).** These explain why papers report 90%+.
12. **Re-runs under the papers' own protocols:** Seeing Red's paper matches; Yadav et al. does not yet; the CWT-CNN protocols are running.

## Model and variants

WaveSpike-PPG has about 5,200–6,300 trainable parameters. It runs causally, one sample at a time, on 50 Hz PPG.

1. **Learnable wavelet:** a causal, undecimated wavelet transform with 5 levels and 6 bands.
    - Each level has its own 8-tap low-pass filter, initialised to Daubechies-4, plus a soft orthonormality loss.
    - The warm-up is 217 samples (4.34 s).
2. **Learnable shrink-only hard threshold**, which sparsifies the spikes.
3. **Per-band gain**, then a ± two-channel LIF spike encoder with 12 spike channels.
4. **Recurrent LIF layer:** 64 neurons, orthogonal recurrent weights, fast-sigmoid surrogate gradient.
5. **Leaky readout**, computed in closed form.
6. **Training:**
    - Loss: class-weighted cross-entropy + wavelet loss + a firing-rate penalty.
    - Optimiser: AdamW (lr 2e-3, wavelet 5e-4), cosine schedule.
    - AF default: at most 30 epochs with early stopping (patience 8).
    - Biometric recipe: 150 epochs, no early stopping.

| Variant | What it changes |
| --- | --- |
| `full` | Proposed model, notebook defaults |
| `irreg_lr` | Not neural: 3 heartbeat-interval features + logistic regression (AF) |
| `gru_twin`, `x_gru_amp` | Non-spiking GRU of matched size on the same front end |
| `snn_raw`, `x_raw_tau_amp` | SNN with no wavelet |
| `fixed_wavelet`, `x_fixed_tau_amp` | Wavelet frozen at Daubechies-4 |
| `x_aug_shared_tau_amp` | Partly learnable wavelet: one learned filter shared by all levels |
| `lean`, `x_lean_tau_amp` | Drops two bands and halves the time steps (energy-saving) |
| `tau_het`, `*_tau_*` | Hidden time constants spread over 0.2–4 s |
| `aug`, `x_aug_*` | Amplitude, shift, noise and baseline-wander augmentation |
| `*_amp` | Adds the window's level and pulse strength to the output |
| `x_pk5_*`, `x_pk4_*` | Wavelet-packet front end, 16 or 8 bands |
| `x_aug_e300`, `x_aug_sdrop`, `x_aug_wavlr`, `x_h128_*` | 300 epochs, spike dropout, faster wavelet learning, 128 neurons |
| AF-only variants | `ht_despawn`, `spkdrop`, `h32`, `augplus`, `slowtrain`, `wavlr`, `raw_tau`, `tau_aug`, `tau_aug_slow`, `lean_tau_aug`, `raw_tau_aug_slow`, `tas_*`, `lean_tas` |
| `beat_lr`, `beat_lr_amp`, `beat_svm_amp` | Not neural: beat-template features + logistic regression or RBF SVM (Seeing Red paper style) |
| `dwt_knn`, `dwt_svm`, `dwt_lda`, `dwt_svm_amp` | Not neural: fixed db8 DWT features + kNN / SVM / shrinkage LDA (generic baselines) |
| `cwt_cnn`, `cwt_cnn_amp` | Fixed Morlet CWT scalogram (32 scales, 0.5–8 Hz) + small 2-D CNN, 60 epochs |

## Datasets and protocols

The data is not in the repository. Download each dataset yourself and respect its licence.

| Dataset | Task | Size used | Licence | Protocol |
| --- | --- | --- | --- | --- |
| MIMIC PERform AF (Zenodo 6807403) | AF; same-session biometrics | 35 subjects, 1,342 windows of 30 s | Check the record (ODbL vs CC BY 4.0) | AF: 5-fold subject-wise, 3 fold seeds × 2 training seeds |
| Zenodo 23597128 (wrist PPG + ECG AF labels) | AF; biometrics across days | AF: 19 subjects, 8,927 windows. Biometrics: 17 subjects, 6,000 windows of 10 s | CC BY-NC-SA 4.0 | Biometrics: session = recording day, rotation folds |
| Seeing Red (Lovisotto et al., CVPRW 2020) | Biometrics across sittings | 14 people, 128 finger-on-camera videos | ODC-PDDL | Sittings = videos less than 1 h apart; test on a later sitting |
| BIDMC (PhysioNet) | Biometrics, same session, clean data | 53 patients, 8 min each | ODC-BY 1.0 | Random window split |
| CinC 2015 (PhysioNet) | Alarm true/false (sanity check) | 383 windows | Open | 5-fold out-of-fold |

**Metrics.**
- **Biometrics** (METRICS_REPORT metrics 1–5):
  - top-1 accuracy and macro-F1;
  - EER;
  - FAR and FRR at a threshold fixed on validation at 2% FAR;
  - AUC and per-person FRR;
  - decisions from 1, 3 or 5 windows.
- **AF:** pooled out-of-fold AUROC, macro-F1, sensitivity, specificity and subject AUROC.

## Results: AF detection

MIMIC PERform AF, 6 repeats, AUROC at the best / last checkpoint:

| Model | AUROC | Verdict vs `full` (best \| last) |
| --- | --- | --- |
| `irreg_lr` | 0.919 / 0.919 | better 6/6 \| better 6/6 |
| `snn_raw` | 0.656 / 0.715 | none \| better 6/6 |
| `spkdrop` | 0.640 / 0.694 | none |
| `aug` | 0.631 / 0.685 | none |
| `fixed_wavelet` | 0.631 / 0.684 | none |
| `ht_despawn` | 0.621 / 0.670 | none |
| `full` | 0.611 / 0.668 | — |
| `lean` | 0.613 / 0.650 | none |
| `h32` | 0.581 / 0.627 | none |
| `gru_twin` | 0.526 / 0.517 | worse 6/6 |

**Other AF results:**
- **One optimistic dev repeat:** `tau_aug_slow` 0.809 and `augplus` 0.777 AUROC.
- **Zenodo wrist AF:** every method scored 0.50–0.57 AUROC.
- **CinC 2015:** 0.672 for `full`, 0.671 for `irreg_lr`.

**Why AF failed:** in MIMIC PERform every window inherits its recording's label, so models can score by recognising the patient. On wrist data, where AF and normal rhythm occur in the same person, that shortcut disappears, and everything is at chance. Possible causes: 50 Hz wrist PPG too noisy, or misaligned labels (not checked).

## Results: biometrics

EER (lower is better) and top-1 accuracy, decision from 5 windows. Mean over 3 folds × 2 seeds; classical models have 3 runs.

| Model | Zenodo across days: EER / top-1 | Seeing Red across sittings: EER / top-1 |
| --- | --- | --- |
| CWT scalogram + CNN | 14.0% / 61.4% | 20.4% / 51.2% |
| Our SNN (`x_tau_amp`; Seeing Red `x_aug_e300`) | 15.6% / 49.2% | 21.8% / 43.3% |
| Our SNN, lean | 16.9% / 46.9% | 25.5% / 40.4% |
| Same-size GRU | 18.3% / 45.2% | 21.7% / 43.6% |
| DWT + SVM | 19.3% / 44.8% | 21.0% / 49.7% |
| Beat features + SVM | 19.7% / 39.5% | 19.6% / 50.5% |
| SNN without wavelet | 20.2% / 30.2% | 29.5% / 37.5% |

**Repeat-by-repeat checks on Zenodo** (our SNN's wins out of 6):

| Our SNN vs | 1 window | 5 windows |
| --- | --- | --- |
| SNN without wavelet | 6/6 | 6/6 |
| Same-size GRU | 6/6 | 6/6 |
| Beat-feature SVM | 4/6 | 6/6 |
| CWT-CNN | 0/6 | 1/6 |

**Wavelet tiers on Zenodo** (5 windows):

| Front end | EER |
| --- | --- |
| No wavelet | 20.2% |
| Fixed | 16.1% |
| Partly learnable | 16.1% |
| Fully learnable | 15.6% |

**Clean data, same session.** Top-1 accuracy from 1, 3 and 5 windows; all classical runs are complete:

| Model | BIDMC (53 people) | MIMIC (35 people) |
| --- | --- | --- |
| Beat features + SVM | 95.8 / 99.8 / 100% | 95.7 / 99.6 / 100% |
| DWT + SVM | 94.6 / 99.6 / 100% | 95.1 / 99.4 / 100% |
| CWT scalogram + CNN | 94.1 / 99.4 / 100% | 93.6 / 98.7 / 99.6% |

A "100%" here means 85–136 test decisions per split, all correct. Test windows come from the same recording as the training windows, so this measures recognising a recording, not a person on another day.

On the wearable data, even the same-session split stays below 75%. Signal quality is the main cause of the gap to the literature.

**Reproducing papers under their own protocols** (`biometrics/paper_repro.py`):

| Paper | Reported | Ours | Verdict |
| --- | --- | --- | --- |
| Lovisotto et al. 2020 (Seeing Red) | EER about 15% (1 beat), < 5% (> 10 beats) | 9.7% (1 beat), 2.2% (10 beats); top-1 73% → 96% | Matches |
| Yadav et al. 2017 (CWT + DLDA, CapnoBase) | EER 0.46% | BIDMC 12.2%, MIMIC 11.6% (all test data) | Does not match yet; pipeline differences being fixed |
| CVT-ConvMixer (CWT + CNN, BIDMC/MIMIC) | 95% accuracy | Running | Pending |
| Low-frame-rate PPG (CFISHR, not open) | 98% accuracy | Running on Seeing Red | Pending |

**Training status:**
- **Fully trained:** every main biometric neural result (150 epochs, cosine to zero), the CWT-CNN (60 epochs) and all classical models.
- **Stopped early:** the AF results and the early biometric screens, through the notebook's early stopping. This is detailed per model in `docs/results.html`.

## Literature

- **Novelty.** No published paper was found that uses an SNN or a learnable wavelet for PPG biometrics. That is the novelty claim; do one more search before submission.
- **Comparability.** Published numbers mostly come from same-session tests on clean data, so they are context only.
- **Yang et al. 2020** uses sparse softmax vectors + kNN. It is not a wavelet method; our `dwt_knn` is a generic baseline, not their method.

## Edge deployment and energy

- **`edge/rpi_infer.py`:** NumPy-only inference, exported by the notebook. It agrees 100% with PyTorch on all 9 exportable configurations. The full model file is 26.5 kB.
- **Latency on a desktop CPU** (not the Pi): 17 ms per 30 s window for `full`, 9 ms for `lean`.
- **Sparsity:** about 10% of input and hidden neurons fire. That is about 543,000 event-driven additions per 30 s window, against 7.5 million dense.
- **`edge/pi_bench.py`:** measures latency percentiles, peak RAM, model size, temperature and clock state, and energy from an external meter log.
- **Caveat:** on a Pi CPU, spikes save nothing, because NumPy does dense arithmetic. Real savings need event-driven hardware or a zero-skipping kernel; otherwise report SynOps × energy per operation as an estimate.

## Repository layout and how to run

| Path | What it is |
| --- | --- |
| `notebook/wavespike_ppg_colab_v3.ipynb` | AF pipeline: data, model, training, evaluation, NumPy export |
| `notebook/cells/` | The notebook's source cells; `biometrics/bio_run.py` imports the model from them |
| `biometrics/prepare_seeing_red.py` | Turns Seeing Red videos into PPG (`raw/seeing_red/sr_rgb50.npz`) |
| `biometrics/prepare_zenodo.py` | Builds the across-days Zenodo windows (`processed/zenodo_bio_50hz_10s.npz`) |
| `biometrics/bio_run.py` | Biometric runner: datasets, splits, baselines, the SNN, metrics |
| `biometrics/paper_repro.py` | Re-runs published methods under their own protocols |
| `edge/pi_bench.py`, `edge/rpi_infer.py` | Raspberry Pi benchmark and NumPy inference |
| `docs/results.html` | Every result and metric (open in a browser) |
| `manuscript/` | Manuscript drafts (Word) |

```bash
export WAVESPIKE_DATA_ROOT=$PWD/wavespike_data      # raw/ and processed/ datasets live here; results/ is written here
BIO_CHANNEL=luma python -I biometrics/bio_run.py --dataset seeing_red --variants beat_svm_amp,x_aug_tau_amp --train-stride 0.5 --threads 12
BIO_CHANNEL=luma python -I biometrics/bio_run.py --dataset seeing_red --train-stride 0.5 --summary --variants beat_svm_amp,x_aug_tau_amp
python -I biometrics/bio_run.py --dataset bidmc_bio --split random --variants beat_svm_amp     # same-session protocol
python -I biometrics/paper_repro.py $WAVESPIKE_DATA_ROOT lovisotto lovisotto.json
```

The requirements are Python 3.13 or newer, CPU PyTorch, NumPy, SciPy, scikit-learn and pandas, plus h5py, PyWavelets and opencv-python-headless for the biometric data. On CPU, use one process with about 12 threads.

## Known issues and limitations

- **Seeing Red:** use luma; the red channel is saturated.
- **Model selection:** validation loss picks bad checkpoints on biometrics, which is why the recipe trains a fixed 150 epochs.
- **Small cohorts** (14–53 people), so confidence intervals are wide.
- **Optimistic pick:** the Seeing Red "best SNN" (`x_aug_e300`) was chosen on test.
- **Closed-set task:** no test on unseen people; a metric-learning loss would allow one.
- **Licence:** the Zenodo data is share-alike, so check whether trained weights count as derived data.

## Open tasks

- [ ] Run the Pi benchmark (needs Pi access and a power meter); export a biometric model; port the CWT-CNN and SVM for comparison.
- [ ] Finish the Yadav et al. reproduction with their exact pipeline: peak-centred segments, one Morse scale, Pearson matching.
- [ ] Finish the same-session neural runs and the CWT-CNN paper protocols.
- [ ] Rewrite the manuscript around biometrics plus edge efficiency, with AF as a negative result.
- [ ] Verify references and literature figures; resolve the MIMIC PERform licence.
- [ ] Inspect the learned wavelet filters.
