# WaveSpike-PPG experiment log

This is the running record of every experiment: what was run, why, with which script and settings, where the results are and what they showed. Add an entry whenever a run starts or finishes, so no work is lost. Numbers are from our own runs unless marked **(paper)**.

Result files live on the compute machine under `wavespike_data/results/` (not in git). Each run writes one JSON file per fold and seed with every metric and every test probability, so any statistic can be recomputed without retraining.

## Model naming rule

Every model is written and trained by us. When a model follows a paper, its name says so and how closely:

- **Ours**: this project's own design.
  - Our SNN, with the wavelet type always stated: fully learnable (`x_tau_amp`), partly learnable (`x_shared_tau_amp`), fixed (`x_fixed_tau_amp`), none (`x_raw_tau_amp`).
  - Lean SNN (`x_lean_tau_amp`).
  - GRU control (`x_gru_amp`).
  - **Our CWT-CNN** (`cwt_cnn_amp`, `cwt_cnn_x_amp`): our own small CNN on a Morlet scalogram, in the style of the CWT-CNN papers, **not** any paper's exact network.
- **Ours, paper-style**: our own feature sets with standard classifiers.
  - Beat features + SVM/LR (`beat_*`), in the style of Lovisotto et al. 2020.
  - DWT features + SVM/LDA/kNN (`dwt_*`), generic baselines.
- **Paper reproduction**: a paper's own method, built from its text. Every gap the paper leaves is listed in the script and in the result file.
  - Lovisotto 2020 and Yadav 2017: `biometrics/paper_repro.py`.
  - Ibrahim 2024 and Rahman 2025: `biometrics/paper_models.py`.
  - P1–P7: planned below.

## Log

### Session-disjoint biometrics (main results), done
- Zenodo wrist PPG across days (17 people), 3 rotation folds × 2 seeds. EER, 5 windows:
  - our CWT-CNN 14.0%;
  - our SNN with fully learnable wavelet 15.6%;
  - our GRU 18.3%;
  - our beat SVM 19.7%;
  - our SNN without wavelet 20.2%.
  - The SNN beats the SVM, GRU and no-wavelet SNN in 6 of 6 repeats, and loses to the CNN in 5 of 6.
- Seeing Red across sittings (14 people): our beat SVM 19.6%; our best SNN 21.8% (`x_aug_e300`, picked on test, so optimistic; picked on validation, `x_aug_wavlr`, 25.4%).

### Same-session protocol (as most papers use), mostly done
Results are top-1 accuracy at 1 · 3 · 5 windows.

| Model | BIDMC | MIMIC |
|---|---|---|
| Our beat SVM | 95.8 · 99.8 · 100 | 95.7 · 99.6 · 100 |
| Our CWT-CNN | 93.7 · 99.7 · 100 | 93.6 · 98.7 · 99.6 |
| Our SNN, fully learnable wavelet, with augmentation | 90.6 · 97.3 · 99.2 | 86.0 · 95.8 · 98.2 |
| Our SNN, fully learnable wavelet | 84.9 · 96.6 · 98.2 | 83.8 · 95.4 · 98.3 |
| Lean SNN | 83.1 · 95.6 · 97.6 | 80.3 · 93.1 · 96.8 |
| GRU | 80.0 · 93.3 · 98.4 | 73.3 · 89.6 · 95.1 |
| SNN without wavelet | 66.3 · 86.9 · 91.8 | 73.3 · 89.7 · 95.9 |

- Zenodo, same session: our CWT-CNN 73.8% and our SNN 54.6% (5 windows).
- Still running: Zenodo same-session GRU (one rerun; its result file had been left empty by the reboot), plus the augmented, no-wavelet and lean SNNs.

### Wavelet ablation, partly running
EER %, 5 windows, 6 repeats. Only the wavelet changes within a column.

| Front end | Zenodo, no aug | Zenodo, aug | Seeing Red, no aug | Seeing Red, aug |
|---|---|---|---|---|
| None | 20.2 | running | 29.5 | running |
| Fixed db4 | 16.1 | running | 24.7 | 23.8 |
| Partly learnable (one shared filter) | running | 16.1 | running | 23.9 |
| Fully learnable (filter per level) | 15.6 | 16.0 | 28.7 (3 of 6 runs) | 24.3 |

- The earlier page compared rows with and without augmentation; this table fixes that.
- New variants added to fill the gaps: `x_shared_tau_amp`, `x_aug_raw_tau_amp`.
- Screens: `wA` (Zenodo), `wB` (Seeing Red).

### Our CWT-CNN trained with exactly our SNN's recipe, running
- What changes: `cwt_cnn_x_amp` gets 150 epochs × 1,600 class-balanced windows, the same augmentation (applied before the CWT), lr 2e-3, batch 64 and gradient clipping 1.0, with the last epoch kept.
- Why: the CNN beat the SNN with a shorter recipe (60 epochs, no augmentation), so this checks the comparison is fair.
- Where: Zenodo across days (screen `cA`) and Seeing Red (screen `cB`).
- **Result (done, 6/6 runs on each dataset):** EER %, last epoch; top-1 in brackets.

| Model | Zenodo, 1 window | Zenodo, 5 windows | Seeing Red, 1 window | Seeing Red, 5 windows |
|---|---|---|---|---|
| Our CWT-CNN, its own recipe (`cwt_cnn_amp`) | 20.3 (39.2%) | 14.0 (61.4%) | 21.4 (49.6%) | 20.4 (51.2%) |
| **Our CWT-CNN, SNN's recipe (`cwt_cnn_x_amp`)** | 22.0 (34.3%) | 15.8 (48.3%) | 21.1 (47.7%) | 20.0 (49.4%) |
| Our SNN, fully learnable wavelet, same recipe (`x_aug_tau_amp`) | 22.5 (31.2%) | 16.0 (43.9%) | 26.7 (36.5%) | 24.3 (37.5%) |

- Zenodo: the SNN's recipe makes the CNN **worse** (14.0 → 15.8% EER; top-1 61.4 → 48.3%). Under identical training, the SNN and CNN are close on average (16.0 vs 15.8%), but the CNN still wins 4 of 6 repeats, so by the all-repeats rule this is not a tie claim.
- Seeing Red: the CNN wins 6 of 6 under either recipe.
- Conclusion: the CNN's lead on Zenodo came partly from its own (shorter, unaugmented) recipe, not only from its architecture. Each model does best with its own recipe; neither recipe makes the SNN win.

### Paper reproductions
- **Lovisotto 2020**, on its own data and protocol: matches. EER 9.7% (1 beat) and 2.1% (20 beats); top-1 73% → 96%.
- **Yadav 2017**: our version is approximate and does not match. 12.2% on BIDMC vs **(paper)** 0.46% on CapnoBase. Their exact pipeline (peak-centred segments, single Morse scale, DLDA, Pearson matching) is still to build.
- **CVT-ConvMixer protocol with our own CNN**: BIDMC 93.8% random / 92.2% chronological; MIMIC 88.2% / 81.3%. **(paper)** 95%.
- **Low-frame-rate protocol with our own CNN**: Seeing Red 50.8% / 43.0%. **(paper)** 98% on CFISHR.
- **Their exact networks** (`biometrics/paper_models.py`, running):
  - Ibrahim et al. 2024 CVT-ConvMixer (22.5M parameters) on RW-PPG, one of their own three datasets. RW-PPG has been downloaded (Mendeley 10.17632/yynb8t9x3d, CERN OHL licence). Screen `pmA`, 5 folds; about 2 min per epoch while the CPU is shared.
  - Rahman & Banavar 2025 CVT-ConvMixer-LSTM on BIDMC, and on Seeing Red decimated to 14 Hz (screen `pmB`).
  - Rahman (BIDMC, then Seeing Red at 14 Hz) was set to the lowest CPU priority (`nice` 19) at the user's request. Under contention it took about 29 min per epoch, so it now uses only spare CPU and finishes after the other jobs.
  - Ibrahim on RW-PPG runs all 5 folds. The user first chose 1 fold, then changed this to all 5, run after the other jobs. It now runs at the lowest CPU priority (`nice` 19), so it uses only spare CPU; the other jobs take priority.
  - CFISHR is on IEEE DataPort under a signed licence (EULA), so it is not open access and is not used.
  - Ibrahim on BIDMC/MIMIC would take about 45 h each on our CPU and has not been started.

### Energy without hardware, running
- `edge/energy_estimate.py` (screen `eA`) counts per 10 s window:
  - parameters;
  - dense multiply-adds;
  - spike-driven adds (SynOps), using spike rates measured on test windows;
  - neuron updates.
- It then applies four energy models:
  - **E1:** 45 nm fp32 (Horowitz 2014, Fig. 1.1.9: multiply-add 4.6 pJ, add 0.9 pJ);
  - **E2:** int8 (0.23 / 0.03 pJ);
  - **E3:** E1 plus SRAM weight reads at 5 pJ per weight;
  - **E4:** Loihi (Davies 2018: 23.6 pJ per synaptic op, 81/52 pJ per active/inactive neuron update, 1.7 pJ per spike).
- The SNN is scored both dense (as a CPU runs it) and event-driven.
- Output: `results/energy_estimates.json`, to be compared later with energy measured on the Pi.
- Both per-operation tables were checked against the source PDFs.
- A smoke test gave, for our SNN: dense E1 14.3 µJ, event-driven E1 0.73 µJ, Loihi 6.6 µJ per window. These come from a 1-epoch model, so they are not results.
- The CPU energy counters (RAPL) on the workstation need root to read; not enabled.

#### Energy estimates, done
- Zenodo across days, fold 0, seed 0, per 10 s window. Saved in `results/energy_estimates.json`.
- Spike rates were measured on the test windows of a model retrained with the same configuration. That model is not bit-identical to the stored run: it agrees on 79–91% of test decisions, because thread counts differ.

| Model | Params | Dense multiply-adds | fp32 dense (µJ) | fp32 event-driven (µJ) | int8 dense / event (µJ) | + SRAM reads, dense / event (µJ) | Loihi (µJ) |
|---|---|---|---|---|---|---|---|
| Our SNN, fully learnable wavelet (`x_tau_amp`) | 6,241 | 3.07 M | 14.3 | 0.75 | 0.71 / 0.033 | 29.1 / 1.80 | 7.3 |
| Same, with augmentation | 6,241 | 3.07 M | 14.3 | 0.67 | 0.71 / 0.031 | 29.1 / 1.26 | 5.1 |
| Lean SNN | 5,973 | 1.49 M | 6.9 | 0.47 | 0.35 / 0.022 | 14.1 / 0.96 | 3.5 |
| Our GRU | 6,288 | 2.73 M | 13.0 | – | 0.65 | 26.1 | – |
| Our CWT-CNN | 24,659 | 11.7 M (incl. CWT) | 54.0 | – | 2.7 | 54.2 | – |
| Our beat SVM | 300k stored values (support vectors) | 0.36 M | 1.7 | – | 0.08 | 3.2 | – |

- Measured spike rates: encoder 3.8–5.5%, hidden layer 4–7%.
- Reading the table:
  - Dense, the SNN costs about 4× less than the CNN.
  - Event-driven, the SNN costs about 70× less than the CNN, but only on hardware that skips silent neurons.
  - Loihi's per-operation costs are higher than the 45 nm figures. The SVM's feature extraction is not counted.
- CPU latency was measured while the machine was overloaded, so it is not reported.

### AF detection (stopped)
- MIMIC PERform: our SNN AUROC 0.611 (last) / 0.668 (best), against the interval baseline 0.919.
- Zenodo wrist: every method at chance.
- CinC 2015: 0.672 vs 0.671.

### P1–P7 spiking-wavelet designs, running (started after the go-ahead)
- Code: `biometrics/spiking_wavelets.py`. Every gap a paper leaves, and every choice made to fill it, is in the docstring.
- Variants are defined in `bio_run.py`.
- All 17 variants passed a 2-epoch smoke test.

| Code | Design |
|---|---|
| `x_p6_haar`, `x_p6_db2`, `x_p6_db8`, `x_p6_sym4`, `x_p6_coif1` | P6: fixed DWT, family sweep; fixed db4 is `x_fixed_tau_amp` |
| `x_p7_dwtc` | P7 as published: bior1.1 + haar + db1 with learnable scalars |
| `x_p7_mix` | Our P7 extension: db4 + sym4 + coif1 |
| `x_p1_doe` | P1, fixed difference-of-exponentials filters |
| `x_p1_doe_learn` | Our extension: learnable time constants |
| `x_p1_doe_penc` | P1 plus its own fixed encoder (threshold 0.1) |
| `x_p2_dot` | P2, cascaded integrators |
| `x_p4_sep` | P4: Haar + learnable separable conv per sub-band |
| `x_stf_ssa` | Shared 1-D spiking-transformer backbone: Spikformer attention baseline |
| `x_stf_fw` | P7 FW head, parameter-free Haar |
| `x_stf_fwc` | P7 DWT-C |
| `x_stf_swsa` | P3 SWSA |
| `x_stf_fatm` | P5 FATM with ternary spikes |

- Queue: 34 jobs (17 variants × Zenodo across days and Seeing Red across sittings, 3 folds × 2 seeds each).
- Workers: 8 queue workers × 3 threads (`pw_worker.sh`, `pw_jobs.txt`, logs `pw1.log`–`pw8.log` on the compute machine).
- Each worker takes the next job as soon as it finishes, so all cores stay busy once the earlier jobs end.
- **Restart, day 1 about 12:06:** the P6 haar, db2 and db8 runs (both datasets) were stuck in thread contention.
  - Symptoms: about 0.77 core each, about 5× more forced context switches, and no run saved after 5.5 h, while sym4 (same code, started together) saved 6 per dataset at about 35–40 min per run.
  - Action, with the user's approval: stopped them (nothing saved, nothing lost) and requeued them at the front with `--threads 2`.
  - Done so far: P6 sym4 6/6 on both datasets; coif1 1/6 on each.
- **13:30 correction:** the two extra workers inherited the lowered priority from the command that started them, so P7 ran at `nice` 19, the lowest. A normal user cannot raise a priority again, so both P7 runs (27 min in, nothing saved) were stopped and restarted from a fresh shell at `nice` 10.
- **Fidelity checks on the papers' own data (approved; run at the very end, after every other job):**
  - P1/P2: reconstruct MIT-BIH ECG with our DoE and DoT filterbanks and compare the RMSE with their Table II (for example, DoE 0.085 and DoT 0.059 at c = 2, K = 8).
  - P2: their synthetic sine-mixture reconstruction (qualitative only; the paper reports no numbers).
  - P6: MNIST with a DWT front end and a convolutional SNN, compared with their 99.25%.
  - The other four papers' tasks (ImageNet/CIFAR, image restoration, licensed EEG) are not feasible here; they are labelled "rebuilt from the paper's text".
- **13:00:** at the user's request, the running P6 jobs (coif1, haar, db2, db8) were lowered to `nice` 15, and two extra queue workers (`pw9`, `pw10`) were started at the normal level (`nice` 10). The remaining designs, starting with P7, can now run first.

## Plan: partly learnable spiking-wavelet designs P1–P7 (approved; being run)

Goal: compare these designs from the literature with our fully learnable wavelet SNN and our partly learnable (shared-filter) SNN.

None of the seven papers is about PPG:
- P1 is ECG/speech reconstruction;
- P2 is a theory paper with synthetic signals;
- P3 is EEG;
- P4 is image restoration;
- P5 and P7 are image and event-camera classification;
- P6 is MNIST digits.

So each is reproduced as its wavelet design, rebuilt for 1-D PPG, inside the same pipeline: same data, folds, seeds, metrics, training recipe (150 epochs, last epoch) and SNN core where possible. Within a comparison only the wavelet part changes. Every choice a paper leaves open is written in the code and in the result file.

| # | Paper | Wavelet design (diagram tier) | 1-D PPG version | Learnable part |
|---|---|---|---|---|
| P1 | Pedersen, Lindeberg, Gerstoft, arXiv 2605.09770 (2026), *Encoding and decoding temporal signals with spiking bandpass wavelets* | Integrated: the LIF membrane is the filter. Difference of truncated exponentials (DoE), one integrator per scale. Dual ± populations, threshold-crossing coding, fixed threshold (θ = 0.1) | Replace our SWT with K DoE channels ψ_k = h(μ_k) − h(μ_{k−1}), μ geometric with c = 2, covering about 0.4–12 Hz at 50 Hz. Our ± LIF encoder (12 channels) is already their dual-population coding | None (paper is closed-form); optional variant with learnable μ_k |
| P1/P2 DoT | Same, plus Pedersen et al., arXiv 2602.02020 (ICASSP 2026), *Scale-covariant spiking wavelets* | Cascaded LIF integrators per scale (time-causal limit kernel); difference of adjacent-scale kernels | K = 7 cascaded exponentials, μ_1 = c^{1−K}√τ_max, μ_k = c^{k−K−1}√(c²−1)√τ_max, c = √2; bandpass = adjacent-scale difference | None (fixed) |
| P3 | Yuan, Wei, Liu, Front. Neurosci. 2025, *Spiking neural networks for EEG signal analysis using wavelet transform* (SpikeWavformer) | Haar DWT fused with spiking self-attention by concatenation (SWSA) | Small 1-D spiking transformer over time tokens: spiking self-attention branch and Haar DWT → conv → IDWT branch, concatenated. LIF threshold 0.5, 4 time steps | Attention and conv; wavelet fixed |
| P4 | Zhao et al., arXiv 2606.18644 (Pattern Recognition 2026), *Spiking pyramid wavelet transformation* | Fixed Haar DWT + small learnable separable conv per sub-band; pyramid levels merged through IDWT linearity | Haar a-trous pyramid (2 levels per block), spiking depthwise + pointwise conv on each sub-band, outputs summed via IDWT | Small kernels per sub-band; wavelet fixed |
| P5 | Fang et al., ECCV 2024, *Spiking Wavelet Transformer* (SWformer; code: github.com/bic-L/Spiking-Wavelet-Transformer) | Frequency-aware token mixer: spiking Haar → learnable block-diagonal weights (K blocks) → inverse Haar, in parallel with spatial conv and channel-mixing branches, summed. Ternary spikes {−1, 0, 1} | 1-D FATM over time: Haar of the spike train, block-diagonal weights (K = 2 and 4), inverse Haar, plus conv and pointwise branches; ternary IF neuron inside the wavelet branch | Block weights; wavelet fixed |
| P6 | Aghabarar, Kiani, Keshavarzi, JAIDM 11(2) 2023, *Digit recognition in SNNs using wavelet transform* | Two-stage: fixed DWT preprocessing, then constant-current LIF encoding, then conv SNN; several wavelet families compared | Fixed DWT front end with a family sweep (haar, db2, db4, db8, sym4, coif1) into our SNN | None |
| P7 | Wang, Zhang, Zhang, Xu, arXiv 2403.18228 (2024), *Fourier or wavelet bases as counterpart self-attention in Spikformer* (FWformer) | Fixed bases replace self-attention, parameter-free; DWT-C mixes Bior1.1 + Haar + Db1 with three learnable scalars | (a) as published: three bases with learnable a, b, c; (b) our extension with genuinely different bases (db4, sym4, coif1), because **Bior1.1, Haar and Db1 are the same Haar filter**, so (a) reduces to one learnable gain | Mixing scalars only |

Order, cheapest and closest to our model first:
1. P6 and P7: front-end swaps only.
2. P1 and P2: replace the SWT with the LIF filterbank.
3. P4: per-sub-band kernels.
4. P5 and P3: these need a small 1-D spiking transformer core, so they also get a version with our recurrent core to isolate the wavelet effect.

Runs: Zenodo across days and Seeing Red across sittings, 3 folds × 2 seeds, both with and without augmentation. Energy is counted for each with `edge/energy_estimate.py`.

## Open items
- RW-PPG licence (CERN OHL) is an unusual licence for data. Check before publishing results that use it.
- Enabling the CPU energy counters (RAPL) needs root on the workstation: the owner's decision.
- Ibrahim network on BIDMC/MIMIC (about 45 h each): run or skip?
- A CNN shrunk to the SNN's size (`cwt_cnn` with width 4, about 6k parameters): proposed in the plan, not started.
