#!/usr/bin/env python3
"""WaveSpike-PPG for PPG biometrics (closed-set identification + verification), session-disjoint evaluation.

Reuses the notebook's model and training code unchanged (cells 02, 03, 11, 13, 15, 17 are exec'd from notebook/cells).
New here: Seeing Red / Zenodo-wrist identity loaders, session-disjoint folds, a classical beat-template baseline,
and biometric metrics (top-1, macro-F1, EER, FAR/FRR at a validation-frozen threshold, per-identity FRR).

Usage:
  python -I biometrics/bio_run.py --dataset seeing_red --variants beat_lr,full,snn_raw --folds 0,1,2 --seeds 0,1 [--threads 12]
  python -I biometrics/bio_run.py --dataset seeing_red --summary [--variants ...]
Results: $WAVESPIKE_DATA_ROOT/results/bio_<dataset>_w<win>/<variant>_f<fold>_ts<seed>.json
"""
import argparse, glob, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
# notebook cells: ../notebook/cells in the repository layout, or ./cells next to this file
CELLS = next((p for p in (os.path.join(HERE, "..", "notebook", "cells"), os.path.join(HERE, "cells")) if os.path.isdir(p)),
             os.path.join(HERE, "cells"))

# ------------------------------------------------------------------ notebook code
def load_notebook_code():
    g = {"__name__": "wavespike_bio"}
    def run(cell, stop=None):
        src = open(os.path.join(CELLS, cell + ".py"), encoding="utf-8").read()
        if stop: src = src[:src.index(stop)]
        exec(compile(src, f"cell{cell}", "exec"), g)
    for c, stop in (("02", None), ("03", None), ("11", None), ("13", "m = WaveSpikeNet(CFG)"), ("15", None), ("17", "RES_DIR")):
        run(c, stop)
    src17 = open(os.path.join(CELLS, "17.py"), encoding="utf-8").read()     # variant_config, verbatim from cell 17
    a = src17.index("def variant_config"); b = src17.index("def run_path")
    exec(compile(src17[a:b], "cell17_variant_config", "exec"), g)
    return g

# extra variants for this task (notebook defaults untouched; everything else comes from cells/17.py VARIANTS)
BIO_VARIANTS = {
    "beat_lr": {"_kind": "beat_lr"},                 # classical: mean beat template + derivative + IBI stats -> logistic regression
    "beat_lr_amp": {"_kind": "beat_lr", "_amp": True},   # + absolute level and pulse amplitude (information the z-scored models never see)
    "beat_svm_amp": {"_kind": "beat_lr", "_amp": True, "_svm": True},   # RBF SVM as in Lovisotto et al. 2020 (their best classifier)
}
# accuracy round (v3.2): no early stopping (val loss proved a poor selector), 150 epochs over a fixed 1,600-window
# random subsample per epoch, optional amplitude fusion (logits += Linear([level, log pulse std]), zero-initialised)
_X = {"epochs": 150, "patience": 150, "max_windows_per_epoch": 1600}
BIO_VARIANTS.update({
    "x_tau":          {**_X, "tau_het": True},
    "x_tau_amp":      {**_X, "tau_het": True, "amp_fusion": True},
    "x_h128_tau_amp": {**_X, "tau_het": True, "amp_fusion": True, "hidden": 128},
    "x_aug_tau_amp":  {**_X, "tau_het": True, "amp_fusion": True, "augment": True},
    "x_raw_tau_amp":  {**_X, "tau_het": True, "amp_fusion": True, "use_wavelet": False, "use_ht": False},
    "x_fixed_tau_amp": {**_X, "tau_het": True, "amp_fusion": True, "learn_wavelet": False},
    "x_lean_tau_amp": {**_X, "tau_het": True, "amp_fusion": True, "drop_bands": [0, 1], "snn_decim": 2},
    "x_gru_amp":      {**_X, "amp_fusion": True, "arch": "gru"},
    # stage 2: augmentation (best regulariser in stage 1) combined with the other front ends / sizes
    "x_aug_fixed_tau_amp": {**_X, "tau_het": True, "amp_fusion": True, "augment": True, "learn_wavelet": False},
    "x_aug_lean_tau_amp":  {**_X, "tau_het": True, "amp_fusion": True, "augment": True, "drop_bands": [0, 1], "snn_decim": 2},
    "x_aug_h128_tau_amp":  {**_X, "tau_het": True, "amp_fusion": True, "augment": True, "hidden": 128},
    "x_aug_gru_amp":       {**_X, "amp_fusion": True, "augment": True, "arch": "gru"},
    # stage 3: regularisation / training length on the best SNN (aug + tau_het + amp). augment_plus is NOT used: its
    # time-warp changes heart rate, which is an identity cue here.
    "x_aug_sdrop":  {**_X, "tau_het": True, "amp_fusion": True, "augment": True, "spike_dropout": 0.15},
    "x_aug_wavlr":  {**_X, "tau_het": True, "amp_fusion": True, "augment": True, "lr_wavelet": 5e-3, "lambda_wav": 0.1},
    "x_aug_e300":   {**_X, "tau_het": True, "amp_fusion": True, "augment": True, "epochs": 300, "patience": 300},
    # wavelet-tier comparison on the same SNN: static (x_aug_fixed_tau_amp) / partially learnable (one shared learnable
    # filter for all levels) / fully adaptive (x_aug_tau_amp: one learnable filter per level, DeSpaWN-style)
    "x_aug_shared_tau_amp": {**_X, "tau_het": True, "amp_fusion": True, "augment": True, "per_level": False},
    # wavelet ablation, filled so every tier exists with AND without augmentation (only the wavelet changes per row)
    "x_shared_tau_amp":     {**_X, "tau_het": True, "amp_fusion": True, "per_level": False},
    "x_aug_raw_tau_amp":    {**_X, "tau_het": True, "amp_fusion": True, "augment": True, "use_wavelet": False, "use_ht": False},
    "x_aug_gru_fixed_amp":  {**_X, "amp_fusion": True, "augment": True, "arch": "gru", "learn_wavelet": False},
})
# richer filterbank: undecimated wavelet-packet front end (make_packet); depth 5 = 32 uniform 0.78 Hz bands, keep <= 12.5 Hz
_PK5 = {"packet_depth": 5, "packet_fmax": 12.5}; _PK4 = {"packet_depth": 4, "packet_fmax": 12.5}
BIO_VARIANTS.update({
    "x_pk5_tau_amp":       {**_X, **_PK5, "tau_het": True, "amp_fusion": True},
    "x_aug_pk5_tau_amp":   {**_X, **_PK5, "tau_het": True, "amp_fusion": True, "augment": True},
    "x_pk5_fixed_tau_amp": {**_X, **_PK5, "tau_het": True, "amp_fusion": True, "learn_wavelet": False},
    "x_pk4_tau_amp":       {**_X, **_PK4, "tau_het": True, "amp_fusion": True},
})
# Published fixed-wavelet biometric pipelines, re-implemented on our data / folds / metrics (see wavelet_features, cwt_cnn):
BIO_VARIANTS.update({
    "dwt_knn":     {"_kind": "beat_lr", "_feat": "dwt", "_clf": "knn"},          # DWT (db8) features + k-NN (generic baseline)
    "dwt_svm":     {"_kind": "beat_lr", "_feat": "dwt", "_svm": True},           # DWT features + RBF SVM
    "dwt_lda":     {"_kind": "beat_lr", "_feat": "dwt", "_clf": "lda"},          # wavelet features + LDA (Yadav/Sarkar CWT+DLDA style)
    "dwt_svm_amp": {"_kind": "beat_lr", "_feat": "dwt", "_svm": True, "_amp": True},
    "cwt_cnn":     {"_kind": "cwt_cnn"},                                         # Morlet scalogram + small 2-D CNN (2023-25 CWT-CNN papers)
    "cwt_cnn_amp": {"_kind": "cwt_cnn", "_amp": True},
    # our CWT-CNN trained with exactly the SNN's (x_aug_tau_amp) recipe: 150 epochs x 1,600 balanced windows, augmentation, lr 2e-3, clip
    "cwt_cnn_x_amp": {"_kind": "cwt_cnn", "_amp": True, "_recipe": True},
})
BIO_VARIANTS_END = {
    "h128":    {"hidden": 128},
    "long":    {"epochs": 100, "patience": 25},
    "tau_long": {"tau_het": True, "epochs": 100, "patience": 25},
    "aug_long": {"augment": True, "epochs": 100, "patience": 25},
}
BIO_VARIANTS.update(BIO_VARIANTS_END)

# ------------------------------------------------------------------ data
def causal_highpass(x, fs, fc=0.5):
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(2, fc, "highpass", fs=fs, output="sos"), x)

def sittings(ts, gap_s=3600):
    """Group one user's measurement timestamps (s) into sittings: a new sitting after a gap > gap_s."""
    o = np.argsort(ts); sid = np.zeros(len(ts), int); k = 0
    for a, b in zip(o[:-1], o[1:]):
        if ts[b] - ts[a] > gap_s: k += 1
        sid[b] = k
    return sid

CHANNELS = {"luma": [0.299, 0.587, 0.114], "red": [1, 0, 0], "green": [0, 1, 0]}     # luma = ITU-R BT.601, as in Lovisotto et al. 2020
AMP = None                                                                            # per window: [mean level, pulse std] before z-scoring
OFF = None                                                                            # per window: start offset (samples) in its recording

def make_amp_fusion(g, n_amp=2):
    """Wrap build_model so cfg.amp_fusion=True models receive windows with n_amp extra trailing columns
    ([level, log pulse std], z-scored on the training fold) and add a zero-initialised linear term to the logits."""
    import torch.nn as nn
    class AmpFusion(nn.Module):
        def __init__(self, base, K):
            super().__init__()
            self.swt = base.swt                    # registered first so the wavelet parameter is named "swt.h0" (train() relies on it)
            self.base, self.cfg, self.w0 = base, base.cfg, getattr(base, "w0", 0)
            self.amp = nn.Linear(n_amp, K); nn.init.zeros_(self.amp.weight); nn.init.zeros_(self.amp.bias)
        def forward(self, x, record=False):
            lg, aux = self.base(x[:, :-n_amp], record) if record else self.base(x[:, :-n_amp])
            return lg + self.amp(x[:, -n_amp:]), aux
    orig_build = g["build_model"]
    g["build_model"] = lambda cfg: AmpFusion(orig_build(cfg), cfg.n_classes) if getattr(cfg, "amp_fusion", False) else orig_build(cfg)
    for name in ("augment_batch", "augment_plus_batch"):          # never augment the appended amplitude columns
        f = g[name]
        def wrapped(xb, rng, fs, _f=f):
            if g.get("AMP_COLS", 0): k = g["AMP_COLS"]; return np.concatenate([_f(np.ascontiguousarray(xb[:, :-k]), rng, fs), xb[:, -k:]], 1).astype(np.float32)
            return _f(xb, rng, fs)
        g[name] = wrapped

def make_packet(g):
    """Optional undecimated wavelet-PACKET front end (cfg.packet_depth > 0): every node of the tree, not only the
    approximation, is split again with the level-j filters (h0_j, QMF h1_j) dilated by 2^j - same causal a-trous
    convolution and warm-up as the notebook's LearnableSWT. Leaves whose (init) response centroid is <= cfg.packet_fmax
    Hz are kept, ordered by frequency; only the tree nodes those leaves need are computed. Default (packet_depth=0)
    leaves the notebook front end untouched."""
    import math, torch, torch.nn as nn, torch.nn.functional as F
    qmf = g["qmf_highpass"]
    class PacketSWT(nn.Module):
        def __init__(s, depth, h0_init, learnable, per_level, fmax, fs):
            super().__init__(); s.levels, s.L, s.per_level = depth, len(h0_init), per_level
            h0 = torch.tensor(np.tile(np.asarray(h0_init, np.float64), (depth if per_level else 1, 1)), dtype=torch.float32)
            if learnable: s.h0 = nn.Parameter(h0)
            else: s.register_buffer("h0", h0)
            s.warmup = sum((2**j)*(s.L - 1) for j in range(depth))
            n = 4096; imp = torch.zeros(1, n); imp[0, 0] = 1.0; f = np.fft.rfftfreq(n, 1/fs)
            with torch.no_grad(): leaves = s._run(imp, None)
            cen = {}
            for p, r in leaves.items():
                R = np.abs(np.fft.rfft(r.reshape(-1).numpy()))**2; cen[p] = float((R*f).sum()/R.sum())
            s.keep = sorted([p for p in cen if cen[p] <= fmax], key=lambda p: cen[p]); s.centroids = [cen[p] for p in s.keep]
            s.need = {p[:k] for p in s.keep for k in range(1, depth + 1)}
        def lowpass(s, j): return s.h0[j if s.per_level else 0]
        def _run(s, x, need):
            paths, a = [()], x.unsqueeze(1)                                  # a: (B, nodes, T)
            for j in range(s.levels):
                h0 = s.lowpass(j); h1 = qmf(h0); d = 2**j
                w = torch.stack([h0, h1]).flip(-1).unsqueeze(1)              # (2, 1, L)
                if need is not None:
                    sel = [i for i, p in enumerate(paths) if p + (0,) in need or p + (1,) in need]
                    a, paths = a[:, sel], [paths[i] for i in sel]
                y_ = F.conv1d(F.pad(a, (d*(s.L - 1), 0)), w.repeat(a.shape[1], 1, 1), dilation=d, groups=a.shape[1])/math.sqrt(2.0)
                a, paths = y_, [p + (c,) for p in paths for c in (0, 1)]
            return {p: a[:, i:i+1] for i, p in enumerate(paths)}
        def forward(s, x):
            nodes = s._run(x, s.need); return torch.cat([nodes[p] for p in s.keep], 1)
    WF = g["WaveFront"]; orig = WF._build_frontend
    def build(self, cfg):
        if not getattr(cfg, "packet_depth", 0): return orig(self, cfg)
        assert cfg.use_wavelet and not cfg.drop_bands
        self.swt = PacketSWT(cfg.packet_depth, g["init_lowpass"](cfg.wavelet_init, cfg.taps), cfg.learn_wavelet, cfg.per_level,
                             cfg.packet_fmax, cfg.fs)
        self.C = len(self.swt.keep); self.keep = list(range(self.C)); self.band_names = [f"{c:.2f}Hz" for c in self.swt.centroids]
        self.decim = int(cfg.snn_decim); self.w0 = -(-self.swt.warmup//self.decim)
        self.ht = g["LearnableHT"](self.C, cfg.ht_alpha, cfg.ht_nonneg, cfg.ht_init) if cfg.use_ht else None
        self.enc_gain = nn.Parameter(torch.ones(self.C))
    WF._build_frontend = build

def load_seeing_red(root, win_s, stride_s, fs=50, skip_s=1.0, channel=None):
    global AMP, OFF
    OFF = []
    channel = channel or os.environ.get("BIO_CHANNEL", "luma")
    d = np.load(f"{root}/raw/seeing_red/sr_rgb50.npz", allow_pickle=True); wch = np.array(CHANNELS[channel])
    users = d["user"].astype(str); ts = d["ts"].astype(float)
    U = sorted(set(users)); uid = {u: i for i, u in enumerate(U)}
    sess = np.zeros(len(users), int)
    for u in U:
        m = users == u; sess[m] = sittings(ts[m])
    X, y, S, V, A = [], [], [], [], []
    n, st, sk = int(win_s*fs), int(stride_s*fs), int(skip_s*fs)
    for i, rgb in enumerate(d["ppg"]):
        lvl = np.asarray(rgb, float) @ wch                 # intensity; more blood -> darker
        x = causal_highpass(-lvl, fs)[sk:]; lvl = lvl[sk:]  # invert so the systolic peak points up
        for a in range(0, len(x) - n + 1, st):
            X.append(x[a:a+n]); y.append(uid[users[i]]); S.append(sess[i]); V.append(i); OFF.append(a)
            A.append([lvl[a:a+n].mean(), np.log(x[a:a+n].std() + 1e-6)])
    AMP = np.asarray(A)
    print("Seeing Red channel:", channel)
    return np.asarray(X, np.float32), np.asarray(y), np.asarray(S), np.asarray(V), U

def load_zenodo_bio(root, win_s, stride_s, fs=50, max_per_day=60):
    """Zenodo wrist PPG (19 subjects, multi-day): identity = subject, session = recording day. Cached by bio_zen_prep.py."""
    d = np.load(f"{root}/processed/zenodo_bio_{fs}hz_{win_s}s.npz", allow_pickle=True)
    return d["X"].astype(np.float32), d["y"], d["sess"], d["vid"], [str(u) for u in d["users"]]

def random_folds(y, n_rot=3, k=5, seed=0):
    """SAME-SESSION protocol (as in most published PPG-biometric papers): windows are split at random, stratified by
    identity, so test windows come from the same recordings and sittings as training windows. Fold r: test = part r,
    val = part r+1, train = the other 3 parts. Used only to show how much the session-disjoint protocol costs."""
    from sklearn.model_selection import StratifiedKFold
    part = np.zeros(len(y), int)
    for i, (_, te) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)): part[te] = i
    return [(np.where((part != r) & (part != (r + 1) % k))[0], np.where(part == (r + 1) % k)[0], np.where(part == r)[0])
            for r in range(n_rot)]

def load_mimic_bio(root, win_s, stride_s, fs=50):
    """MIMIC PERform AF (35 subjects, one ~20 min hospital fingertip PPG recording each) reused for biometrics: identity =
    subject, each cached 30 s window cut into non-overlapping win_s windows. One session per subject, so only the
    same-session (--split random) protocol is possible."""
    d = np.load(f"{root}/processed/mimic_perform_af_50hz_30s.npz", allow_pickle=True)
    subj = d["subject"].astype(str); U = sorted(set(subj)); uid = {u: i for i, u in enumerate(U)}; n = int(win_s*fs)
    X, y, V = [], [], []
    for w, s in zip(d["X"], subj):
        for a in range(0, len(w) - n + 1, n): X.append(w[a:a+n]); y.append(uid[s]); V.append(uid[s])
    y = np.asarray(y); return np.asarray(X, np.float32), y, np.zeros(len(y), int), np.asarray(V), U

def bidmc_signals(root, fs=50):
    """BIDMC (PhysioNet, ODC-BY 1.0): 53 ICU patients, 8 min fingertip PPG (PLETH) at 125 Hz -> 50 Hz. Cached."""
    cache = f"{root}/processed/bidmc_ppg_{fs}hz.npz"
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True); return list(d["sig"]), [str(s) for s in d["subj"]]
    import zipfile, io, pandas as pd
    from scipy.signal import resample_poly
    z = zipfile.ZipFile(f"{root}/raw/bidmc/bidmc.zip"); sig, subj = [], []
    for nm in sorted(n for n in z.namelist() if n.endswith("_Signals.csv") and "bidmc_csv" in n):
        df = pd.read_csv(io.BytesIO(z.read(nm))); df.columns = [c.strip() for c in df.columns]
        x = df["PLETH"].to_numpy(float); x = x[np.isfinite(x)]
        sig.append(resample_poly(x, fs*2, 250).astype(np.float32)); subj.append(os.path.basename(nm).split("_")[1])
    np.savez(cache, sig=np.array(sig, dtype=object), subj=np.array(subj)); return sig, subj

def load_bidmc_bio(root, win_s, stride_s, fs=50):
    """BIDMC for biometrics: identity = patient, non-overlapping win_s windows of the single 8 min recording.
    One session per patient, so only the same-session (--split random) protocol is possible."""
    sig, subj = bidmc_signals(root, fs); n = int(win_s*fs); X, y = [], []
    for i, x in enumerate(sig):
        for a in range(0, len(x) - n + 1, n): X.append(x[a:a+n]); y.append(i)
    y = np.asarray(y); return np.asarray(X, np.float32), y, np.zeros(len(y), int), y.copy(), subj

def session_folds(y, S, n_rot=3, E=None):
    """Fold k: for every identity, test = its (k+1)-th last session, val = the session before that (wrapping), train = the rest.
    Fold 0 is strictly chronological (enrol on the past, test on the latest session)."""
    folds = []
    for k in range(n_rot):
        tr, va, te = [], [], []
        for u in np.unique(y):
            iu = np.where(y == u)[0]; ss = np.unique(S[iu]); n = len(ss)
            assert n >= 3, f"identity {u} has only {n} sessions"
            t = ss[(n - 1 - k) % n]; v = ss[(n - 2 - k) % n]
            te += list(iu[S[iu] == t]); va += list(iu[S[iu] == v]); tr += list(iu[(S[iu] != t) & (S[iu] != v)])
        tr, va, te = np.array(tr), np.array(va), np.array(te)
        if E is not None: va, te = va[E[va]], te[E[te]]          # evaluation keeps only windows on the evaluation stride grid
        folds.append((tr, va, te))
    return folds

# ------------------------------------------------------------------ classical baseline
def beat_features(W, fs):
    from scipy.signal import butter, sosfiltfilt, find_peaks
    sos = butter(2, [0.5, 8], "bandpass", fs=fs, output="sos"); pre, post = int(0.25*fs), int(0.55*fs); F = []
    for w in W:
        x = sosfiltfilt(sos, w); x = (x - x.mean())/(x.std() + 1e-9)
        pk, _ = find_peaks(x, distance=int(0.33*fs), prominence=0.3)
        pk = pk[(pk >= pre) & (pk < len(x) - post)]
        if len(pk) < 2:
            F.append(np.zeros(pre + post + (pre + post - 1) + 3)); continue
        B = np.stack([x[p-pre:p+post] for p in pk]); B = (B - B.mean(1, keepdims=True))/(B.std(1, keepdims=True) + 1e-9)
        t = B.mean(0); ibi = np.diff(pk)/fs
        F.append(np.concatenate([t, np.diff(t), [ibi.mean(), ibi.std(), np.median(B.std(0))]]))
    return np.asarray(F)

def wavelet_features(X, fs, wavelet="db8", level=5):
    """Fixed-wavelet features as in the DWT/CWT PPG-biometric literature: (a) DWT (db8, 5 levels) of the window ->
    per-subband log-energy, std, mean |c|, and the leading approximation coefficients; (b) DWT coefficients of the
    window's mean beat template (beat-level wavelet features as in Liu 2022)."""
    import pywt
    from scipy.signal import butter, sosfiltfilt, find_peaks
    sos = butter(2, [0.5, 8], "bandpass", fs=fs, output="sos"); pre, post = int(0.25*fs), int(0.55*fs); F = []
    for w in X:
        cs = pywt.wavedec(w, wavelet, level=level); f = []
        for c in cs: f += [np.log((c**2).mean() + 1e-9), c.std(), np.abs(c).mean()]
        f += list(cs[0][:16]/(np.abs(cs[0]).max() + 1e-9))
        x = sosfiltfilt(sos, w); pk, _ = find_peaks(x, distance=int(0.33*fs), prominence=0.3*x.std())
        pk = pk[(pk >= pre) & (pk < len(x) - post)]
        if len(pk) >= 1:
            B = np.stack([x[p-pre:p+post] for p in pk]); t = ((B - B.mean(1, keepdims=True))/(B.std(1, keepdims=True) + 1e-9)).mean(0)
        else: t = np.zeros(pre + post)
        f += list(np.concatenate(pywt.wavedec(t, "db4", level=3, mode="periodization")))
        F.append(f)
    return np.nan_to_num(np.asarray(F, np.float64))

def cwt_cnn_probs(X, AMP, y, tr, va, te, K, seed, threads, fs=50, epochs=60, use_amp=False, recipe=None, width=16):
    """OUR OWN CNN (not any paper's exact architecture): Morlet continuous-wavelet scalogram (32 scales, 0.5-8 Hz,
    log-magnitude) -> small 2-D CNN (width/2w/4w channels). Default recipe: class-weighted CE, AdamW 1e-3 + cosine, 60 epochs
    over all training windows, no augmentation, last-epoch weights.
    recipe = the SNN's training recipe (dict with g = notebook globals): 150 epochs, each a class-balanced subsample of
    1,600 training windows, the notebook's augment_batch applied to the raw window BEFORE the CWT, lr 2e-3, batch 64,
    grad-norm clip 1.0, last-epoch weights - i.e. exactly what x_aug_tau_amp gets. No test information is used."""
    import pywt, torch, torch.nn as nn, torch.nn.functional as Fn
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(threads)
    freqs = np.geomspace(0.5, 8, 32); sc = pywt.central_frequency("morl")*fs/freqs
    meth = "fft" if recipe else "conv"          # fft only for the recipe run (CWT recomputed per augmented batch); default unchanged
    def scal(Xb):
        out = np.empty((len(Xb), 32, -(-Xb.shape[1]//4)), np.float32)     # ceil: matches the [::4] slice for any window length
        for i in range(0, len(Xb), 256):
            c, _ = pywt.cwt(Xb[i:i+256], sc, "morl", sampling_period=1/fs, axis=-1, method=meth)      # (32, b, T)
            out[i:i+256] = np.log(np.abs(c[:, :, ::4]).transpose(1, 0, 2) + 1e-3)
        return out
    S = scal(X); mu, sd = S[tr].mean(), S[tr].std() + 1e-6
    S = (S - mu)/sd
    Am = ((AMP - AMP[tr].mean(0))/(AMP[tr].std(0) + 1e-6)).astype(np.float32)
    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            w = width
            s.c = nn.Sequential(nn.Conv2d(1, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(), nn.MaxPool2d(2),
                                nn.Conv2d(w, 2*w, 3, padding=1), nn.BatchNorm2d(2*w), nn.ReLU(), nn.MaxPool2d(2),
                                nn.Conv2d(2*w, 4*w, 3, padding=1), nn.BatchNorm2d(4*w), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                nn.Dropout(0.3))
            s.o = nn.Linear(4*w + (Am.shape[1] if use_amp else 0), K)
        def forward(s, x, a): h = s.c(x.unsqueeze(1)); return s.o(torch.cat([h, a], 1) if use_amp else h)
    m = Net(); opt = torch.optim.AdamW(m.parameters(), 2e-3 if recipe else 1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    cnt = np.bincount(y[tr], minlength=K); cw = torch.tensor(cnt.sum()/(K*np.maximum(cnt, 1)), dtype=torch.float32)
    rng = np.random.default_rng(seed); St, At, yt = torch.tensor(S), torch.tensor(Am), torch.tensor(y)
    if recipe: g = recipe["g"]; g["AMP_COLS"] = 0; arng = np.random.default_rng(seed + 7919)
    for ep in range(epochs):
        m.train()
        if recipe:                               # the SNN's recipe: balanced 1,600-window subsample, augmentation before the CWT
            for b in g["batches"](g["balanced_subsample"](np.asarray(tr), recipe["n_per_epoch"], rng, y), 64):
                xb = g["augment_batch"](np.ascontiguousarray(X[b]), arng, fs)
                sb = torch.tensor((scal(xb) - mu)/sd)
                loss = Fn.cross_entropy(m(sb, At[b]), yt[b], weight=cw); opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        else:
            for b in np.array_split(rng.permutation(tr), max(1, len(tr)//64)):
                loss = Fn.cross_entropy(m(St[b], At[b]), yt[b], weight=cw); opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    m.eval()
    with torch.no_grad():
        P = lambda idx: torch.softmax(torch.cat([m(St[b], At[b]) for b in np.array_split(idx, max(1, len(idx)//256))]), 1).numpy()
        return P(va), P(te), sum(p.numel() for p in m.parameters())

# ------------------------------------------------------------------ biometric metrics
def aggregate(P, y, vid, n):
    """Decision = mean log-probability over n consecutive windows of the same recording (non-overlapping groups of windows)."""
    if n == 1: return P, y
    lp = np.log(np.clip(P, 1e-12, 1)); Po, yo = [], []
    for v in np.unique(vid):
        iv = np.where(vid == v)[0]
        for a in range(0, len(iv) - n + 1, n):
            s = lp[iv[a:a+n]].mean(0); s = np.exp(s - s.max()); Po.append(s/s.sum()); yo.append(y[iv[a]])
    return np.asarray(Po), np.asarray(yo)

def trials(P, y):
    K = P.shape[1]; gen = P[np.arange(len(y)), y]
    mask = np.ones_like(P, bool); mask[np.arange(len(y)), y] = False
    imp = P[mask]; imp_claim = np.tile(np.arange(K), (len(y), 1))[mask]
    return gen, imp, imp_claim

def eer(gen, imp):
    th = np.unique(np.concatenate([gen, imp]))
    far = np.array([(imp >= t).mean() for t in th]); frr = np.array([(gen < t).mean() for t in th])
    d = far - frr; i = np.where(np.diff(np.sign(d)) != 0)[0]
    if len(i) == 0: j = np.argmin(np.abs(d)); return float((far[j] + frr[j])/2)
    i = i[0]; a = d[i]/(d[i] - d[i+1])                          # linear interpolation at the sign change
    return float(((far[i] + a*(far[i+1]-far[i])) + (frr[i] + a*(frr[i+1]-frr[i])))/2)

def tau_at_far(imp, target):
    """Lowest threshold whose FAR (fraction of impostor scores >= tau) is <= target, chosen on VALIDATION."""
    s = np.sort(imp)[::-1]; k = int(np.floor(target*len(s)))
    return float(np.nextafter(s[k], np.inf)) if k < len(s) else float(s[-1])

def bio_metrics(Pv, yv, vv, Pt, yt, vt, n, far_target=0.02):
    from sklearn.metrics import f1_score, roc_auc_score
    Pv, yv = aggregate(Pv, yv, vv, n); Pt, yt = aggregate(Pt, yt, vt, n)
    if len(yt) == 0 or len(yv) == 0:            # fewer than n windows per recording in val/test (e.g. non-overlapping split)
        nan = float("nan")
        return {"n_agg": n, "n_decisions": 0, **{k: nan for k in ("top1", "macro_f1", "eer", "eer_val", "auc", "tau", "far", "frr", "hter")},
                "far_target_val": far_target, "n_genuine": 0, "n_impostor": 0, "frr_per_id": {}}
    K = Pt.shape[1]
    gv, iv_, _ = trials(Pv, yv); gt, it, _ = trials(Pt, yt)
    tau = tau_at_far(iv_, far_target)
    pred = Pt.argmax(1); frr_id = {int(u): float((Pt[yt == u, u] < tau).mean()) for u in np.unique(yt)}
    far, frr = float((it >= tau).mean()), float((gt < tau).mean())
    return {"n_agg": n, "n_decisions": int(len(yt)), "top1": float((pred == yt).mean()),
            "macro_f1": float(f1_score(yt, pred, average="macro", labels=range(K), zero_division=0)),
            "eer": eer(gt, it), "eer_val": eer(gv, iv_), "auc": float(roc_auc_score(np.r_[np.ones(len(gt)), np.zeros(len(it))], np.r_[gt, it])),
            "tau": tau, "far": far, "frr": frr, "hter": (far + frr)/2, "far_target_val": far_target,
            "n_genuine": int(len(gt)), "n_impostor": int(len(it)), "frr_per_id": frr_id}

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="seeing_red"); ap.add_argument("--variants", default="beat_lr,full")
    ap.add_argument("--folds", default="0,1,2"); ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--win", type=float, default=10); ap.add_argument("--stride", type=float, default=2)
    ap.add_argument("--threads", type=int, default=12); ap.add_argument("--summary", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train-stride", type=float, default=None,
                    help="window stride for TRAINING windows (denser = more overlapping windows); val/test keep --stride")
    ap.add_argument("--split", default="session", choices=["session", "random"],
                    help="session = session-disjoint (default); random = same-session random split, for comparison with papers")
    a = ap.parse_args()
    root = os.environ.get("WAVESPIKE_DATA_ROOT") or next(
        (p for p in (os.path.join(HERE, "..", "wavespike_data"), os.path.join(HERE, "wavespike_data")) if os.path.isdir(p)),
        os.path.join(HERE, "..", "wavespike_data"))
    ch = f"_{os.environ.get('BIO_CHANNEL', 'luma')}" if a.dataset == "seeing_red" else ""
    tsx = f"_trs{a.train_stride:g}" if a.train_stride and a.train_stride != a.stride else ""
    sx = "_rand" + (f"_s{a.stride:g}" if a.stride != 2 else "") if a.split == "random" else ""
    res_dir = f"{root}/results/bio_{a.dataset}{ch}_w{a.win:g}{tsx}{sx}"; os.makedirs(res_dir, exist_ok=True)
    variants = a.variants.split(",")
    if a.summary: return summary(res_dir, variants)

    import torch; torch.set_num_threads(a.threads)
    global AMP
    g = load_notebook_code(); g["VARIANTS"].update(BIO_VARIANTS); make_amp_fusion(g); make_packet(g)
    loader = {"seeing_red": load_seeing_red, "zenodo_bio": load_zenodo_bio, "mimic_bio": load_mimic_bio, "bidmc_bio": load_bidmc_bio}[a.dataset]
    trs = a.train_stride or a.stride
    X, y, S, V, U = loader(root, a.win, trs)
    if AMP is None:                                         # datasets without a loader-side level: use the raw window itself
        AMP = np.stack([X.mean(1), np.log(X.std(1) + 1e-6)], 1)
    E = None
    if OFF is not None and trs != a.stride:
        step = int(round(a.stride/trs)); E = (np.asarray(OFF)//int(round(trs*50))) % step == 0
    X = ((X - X.mean(1, keepdims=True))/(X.std(1, keepdims=True) + 1e-6)).astype(np.float32)   # same per-window z-score as cell 09
    CFG = g["CFG"]; CFG.n_classes = len(U); CFG.win_sec = int(round(a.win))
    folds = session_folds(y, S, E=E) if a.split == "session" else random_folds(y)
    print(f"{a.dataset}: X {X.shape} | identities {len(U)} | sessions/identity "
          f"{[len(np.unique(S[y == u])) for u in range(len(U))]} | recordings {len(np.unique(V))}")
    for k, (tr, va, te) in enumerate(folds): print(f"fold {k}: train {len(tr)} val {len(va)} test {len(te)} windows")
    for v in variants:
        spec = g["VARIANTS"][v]
        for k in map(int, a.folds.split(",")):
            for ts in map(int, a.seeds.split(",")):
                p = f"{res_dir}/{v}_f{k}_ts{ts}.json"
                if os.path.exists(p) and not a.smoke: print("skip (done)", p); continue
                if spec.get("_kind") == "beat_lr" and ts != int(a.seeds.split(",")[0]): continue   # deterministic: one seed
                tr, va, te = folds[k]; t0 = time.time(); extra = {}
                if spec.get("_kind") == "beat_lr":
                    from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler
                    from sklearn.pipeline import make_pipeline
                    fk = "FEATS_" + spec.get("_feat", "beat")
                    if fk not in g: g[fk] = wavelet_features(X, CFG.fs) if spec.get("_feat") == "dwt" else beat_features(X, CFG.fs)
                    Fm = g[fk]
                    if spec.get("_amp"): Fm = np.concatenate([Fm, AMP], 1)
                    if spec.get("_svm"):
                        from sklearn.svm import SVC
                        est = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=0)
                    elif spec.get("_clf") == "knn":
                        from sklearn.neighbors import KNeighborsClassifier
                        est = KNeighborsClassifier(n_neighbors=5, weights="distance")
                    elif spec.get("_clf") == "lda":
                        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                        est = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                    else: est = LogisticRegression(C=0.1, max_iter=5000)
                    clf = make_pipeline(StandardScaler(), est).fit(Fm[tr], y[tr])
                    Pv, Pt = clf.predict_proba(Fm[va]), clf.predict_proba(Fm[te]); ent = {"best": (Pv, Pt), "last": (Pv, Pt)}
                elif spec.get("_kind") == "cwt_cnn":
                    rc = {"g": g, "n_per_epoch": 64 if a.smoke else 1600} if spec.get("_recipe") else None
                    ep = 2 if a.smoke else (150 if rc else 60)
                    Pv, Pt, npar = cwt_cnn_probs(X, AMP, y, tr, va, te, len(U), ts, a.threads, CFG.fs,
                                                 epochs=ep, use_amp=spec.get("_amp", False), recipe=rc, width=spec.get("_width", 16))
                    ent = {"best": (Pv, Pt), "last": (Pv, Pt)}; extra = {"params": int(npar), "epochs_run": ep}
                else:
                    c = g["variant_config"](v, ts, base=CFG); c.n_classes = len(U)
                    if a.smoke: c.epochs, c.patience = 2, 1
                    Xa = X
                    if getattr(c, "amp_fusion", False):         # z-score the amplitude features on the TRAINING fold only
                        mu, sd = AMP[tr].mean(0), AMP[tr].std(0) + 1e-6
                        Xa = np.concatenate([X, ((AMP - mu)/sd).astype(np.float32)], 1); g["AMP_COLS"] = AMP.shape[1]
                    else: g["AMP_COLS"] = 0
                    out = g["train"](c, tr, va, verbose=False, Xa=Xa, ya=y)
                    ent = {nm: (g["predict"](out[nm], va, Xa=Xa), g["predict"](out[nm], te, Xa=Xa)) for nm in ("best", "last")}
                    extra = {"epochs_run": len(out["hist"]), "best_epoch": out["best_epoch"],
                             "hid_rate_last": out["hist"][-1]["hid_rate"], "params": int(g["n_params"](out["best"]))}
                rec = {"variant": v, "fold": k, "seed": ts, "dataset": a.dataset, "win": a.win, "stride": a.stride,
                       "minutes": (time.time() - t0)/60, **extra,
                       "y_val": y[va].tolist(), "vid_val": V[va].tolist(), "y_te": y[te].tolist(), "vid_te": V[te].tolist()}
                for nm, (Pv, Pt) in ent.items():
                    rec[nm] = {"p_val": np.round(Pv, 6).tolist(), "p_te": np.round(Pt, 6).tolist(),
                               "metrics": [bio_metrics(Pv, y[va], V[va], Pt, y[te], V[te], n) for n in (1, 3, 5)]}
                if not a.smoke: json.dump(rec, open(p + ".tmp", "w")); os.replace(p + ".tmp", p)
                m = rec["best"]["metrics"][0]; ml = rec["last"]["metrics"][0]
                print(f"{v:12s} f{k} ts{ts} | top1 {m['top1']:.3f}/{ml['top1']:.3f} EER {100*m['eer']:.1f}/{100*ml['eer']:.1f}% "
                      f"| FAR {100*m['far']:.1f}% FRR {100*m['frr']:.1f}% | {rec['minutes']:.1f} min", flush=True)
    print("BIO_DONE")

def summary(res_dir, variants):
    rows = []
    print(f"{res_dir}\nmean over runs (folds x seeds); best/last checkpoint; n = windows per decision")
    print(f"{'variant':16s} runs | n | VAL EER% b/l | top1 b/l    | macroF1 b/l | EER% b/l    | FAR/FRR% @val-tau (last) | EER% range (last)")
    for v in variants:
        R = [json.load(open(p)) for p in sorted(glob.glob(f"{res_dir}/{v}_f*_ts*.json"))]
        if not R: print(f"{v:16s} (no runs)"); continue
        for j, n in enumerate((1, 3, 5)):
            M = {nm: [r[nm]["metrics"][j] for r in R] for nm in ("best", "last")}
            f = lambda nm, key: np.mean([m[key] for m in M[nm]])
            e = [100*m["eer"] for m in M["last"]]
            print(f"{v:16s} {len(R):4d} | {n} | {100*f('best','eer_val'):5.1f}/{100*f('last','eer_val'):5.1f} "
                  f"| {f('best','top1'):.3f}/{f('last','top1'):.3f} | {f('best','macro_f1'):.3f}/{f('last','macro_f1'):.3f} "
                  f"| {100*f('best','eer'):5.1f}/{100*f('last','eer'):5.1f} | {100*f('last','far'):5.1f} / {100*f('last','frr'):5.1f}"
                  f"            | {min(e):.1f}-{max(e):.1f}")

if __name__ == "__main__": main()
