#!/usr/bin/env python3
"""Published CWT-scalogram biometric networks, built as closely as each paper's text allows, run under each paper's own
protocol. Neither paper releases code, and both leave parts of the network unspecified or contradictory; every such gap
is filled with the choice listed in GAPS below (also written into each result file).

  ibrahim   Ibrahim et al., "A Novel PPG-Based Biometric Authentication System Using a Hybrid CVT-ConvMixer Architecture
            with Dense and Self-Attention Layers", Sensors 24(1):15, 2024.  Reported: 95% accuracy over RW-PPG + BIDMC + MIMIC.
  rahman    Rahman & Banavar, "A Hybrid Deep Learning Model for Robust Biometric Authentication from Low-Frame-Rate PPG
            Signals", arXiv 2511.04037, 2025 (CVT-ConvMixer-LSTM).  Reported: 97.68% on CFISHR (46 thumbs, 14 fps;
            IEEE DataPort, EULA - not open), 98% on BIDMC.

Protocols (paper text):
  ibrahim : resample to 250 Hz, band-pass, 5 s segments, 50% overlap, Morlet CWT up to 100 Hz, 5-fold CV with 20% test and
            the rest split 80/20 train/val. Datasets: RW-PPG (open, Mendeley 10.17632/yynb8t9x3d), BIDMC, MIMIC PERform.
  rahman  : band-pass 0.7-4 Hz, Fourier-resample to 70 Hz, min-max, 5 s segments, 50% overlap, Morlet CWT up to 4 Hz,
            256x256 scalogram, 80/20 per-person segment split. Datasets: BIDMC; Seeing Red decimated to 14 Hz as the
            stand-in for CFISHR (both are fingertip camera PPG at a low frame rate).
Each protocol is also run with a chronological 80/20 split (no overlapping segment on both sides of the cut).

Usage: python -I biometrics/paper_models.py <data_root> <model> <dataset> <out.json> [threads] [--smoke]
       model = ibrahim | rahman ; dataset = rwppg | bidmc | mimic | seeing_red14
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.signal import butter, sosfiltfilt, resample, resample_poly
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

GAPS = {
    "ibrahim": [
        "Filter cut-offs not given: 0.5-8 Hz 4th-order Butterworth band-pass.",
        "PCA motion-artifact step: number of components not given; omitted (single-channel PPG).",
        "Normalisation formula not given: per-segment z-score.",
        "CWT scales not given: 64 Morlet scales, log-spaced 1-100 Hz (paper: 'frequency range 0-100 Hz'); magnitude.",
        "Scalogram image size and colour map not given: 128x128, one channel (magnitude, per-image min-max).",
        "CVT feature extractor: '17 conv layers, 3x3, ReLU, BN, five 2x2 max-pools' - channel plan not given: VGG-style "
        "64,64 | 128,128 | 256x4 | 512x4 | 512x5 (17 convs). 128x128 input -> 4x4x512 = 16 tokens, matching Table 3 '16 tokens'.",
        "Transformer: heads 8 (text) vs 4 (Table 3); Table 3 used: 4 heads, embedding 64, 1 block (block count not given).",
        "ConvMixer: Sec. 4.1 values used: patch 5, hidden 32, depth 7, kernel 5.",
        "Fusion: CVT + alpha*ConvMixer with alpha a learnable scalar (init 1); residual; channel attention = GAP->FC(16)->FC(64)->sigmoid.",
        "Classes: Table 3 says 2 (genuine/impostor); one output per person used (identification), as the task describes.",
        "Training: Adam 1e-3 with exponential decay 0.95/epoch, batch 32, dropout 0.2, L2 0.01, 80 epochs (Sec. 4.3), cross-entropy, last epoch kept.",
        "Split: 5-fold CV with the train part split 80/20 train/val (Sec. 4.1).",
    ],
    "rahman": [
        "Detrending method not given: removed by the 0.7-4 Hz band-pass (4th-order Butterworth, zero-phase).",
        "PCA (abstract only, absent from the method): omitted.",
        "Outlier clipping threshold not given: clip at +-3 SD before min-max.",
        "CWT scales not given: 64 Morlet scales, log-spaced 0.5-4 Hz; magnitude resized to 256x256 (stated input size).",
        "CVT conv projection: 'two Conv2D, kernel 7, stride 8, output 32x32x64' - first conv k7 s8 (256->32), second k7 s1; "
        "1024 tokens x 64 (as A_CVT in Sec. II-D1); learnable position embedding; 1 pre-norm transformer block (count not "
        "given), 4 heads, MLP 128.",
        "ConvMixer: 32 filters, kernel 5 (stated); patch size and depth not given: patch 8 (32x32 grid), depth 7 (from the "
        "predecessor paper); MHSA 4 heads on its output (head count not given).",
        "LSTM: unidirectional, 64 units, 1 layer on the raw 1-D segment (equations; the text says bidirectional).",
        "Fusion: concatenation 64+32+64 = 160 -> dense softmax (the '64-d projection' is not described; not added).",
        "Training not given at all: Adam 1e-3, batch 32, 60 epochs, cross-entropy, last epoch kept.",
        "Decision rule: softmax argmax (the cosine-template rule's threshold is not given).",
    ],
}

def bp(x, lo, hi, fs, order=4): return sosfiltfilt(butter(order, [lo, hi], "bandpass", fs=fs, output="sos"), x)

# ------------------------------------------------------------------------------------------------ data
def load(root, ds):
    """-> list of 1-D signals, owner id per signal, sampling rate, people."""
    if ds == "rwppg":
        import pandas as pd
        S, y = [], []
        for f in ("train8", "test8"):                     # rows 0-299 = 6 s at 50 Hz; row 304 = person ID (1..35)
            d = pd.read_excel(f"{root}/raw/rw_ppg/{f}.xlsx", header=None)
            S += [np.asarray(d.iloc[:300, j], float) for j in range(d.shape[1])]; y += [int(d.iloc[304, j]) - 1 for j in range(d.shape[1])]
        return S, np.array(y), 50, len(set(y))
    if ds in ("bidmc", "mimic"):
        from bio_run import bidmc_signals
        if ds == "bidmc": sigs, U = bidmc_signals(root)
        else:
            d = np.load(f"{root}/processed/mimic_perform_af_50hz_30s.npz", allow_pickle=True); s = d["subject"].astype(str)
            U = sorted(set(s)); sigs = [np.concatenate(list(d["X"][s == u])) for u in U]
        return [np.asarray(x, float) for x in sigs], np.arange(len(sigs)), 50, len(sigs)
    if ds == "seeing_red14":                               # 50 Hz luma -> 14 Hz: the CFISHR frame rate
        from bio_run import CHANNELS
        d = np.load(f"{root}/raw/seeing_red/sr_rgb50.npz", allow_pickle=True); w = np.array(CHANNELS["luma"])
        users = d["user"].astype(str); U = sorted(set(users)); uid = {u: i for i, u in enumerate(U)}
        sigs = [resample_poly(-(np.asarray(r, float) @ w)[50:], 7, 25) for r in d["ppg"]]
        return sigs, np.array([uid[u] for u in users]), 14, len(U)
    raise ValueError(ds)

def segments(sigs, owners, fs, model):
    """Paper preprocessing + 5 s / 50% overlap segmentation. RW-PPG signals are 6 s: one 5 s segment each."""
    X, y, rec, t = [], [], [], []
    for r, (s, o) in enumerate(zip(sigs, owners)):
        if model == "ibrahim":
            x = bp(s, 0.5, 8, fs); fo = 250; x = resample_poly(x, fo, fs)
        else:
            x = bp(s, 0.7, min(4, 0.45*fs), fs); fo = 70; x = resample(x, int(round(len(x)*fo/fs)))   # Fourier resampling
        n, h = 5*fo, int(2.5*fo)
        if len(x) < n: x = np.pad(x, (0, n - len(x)))                                    # zero-padding (both papers)
        for a in range(0, len(x) - n + 1, h):
            w = x[a:a+n]
            if model == "ibrahim": w = (w - w.mean())/(w.std() + 1e-9)
            else: w = np.clip(w, w.mean() - 3*w.std(), w.mean() + 3*w.std()); w = (w - w.min())/(w.max() - w.min() + 1e-9)
            X.append(w); y.append(o); rec.append(r); t.append(a)
    return np.asarray(X, np.float32), np.asarray(y), np.asarray(rec), np.asarray(t), fo

def scalograms(X, fs, model):
    import pywt, torch, torch.nn.functional as F
    lo, hi, side = (1.0, 100.0, 128) if model == "ibrahim" else (0.5, 4.0, 256)
    fr = np.geomspace(lo, hi, 64); sc = pywt.central_frequency("morl")*fs/fr
    out = np.empty((len(X), side, side), np.float32)
    for i in range(0, len(X), 128):
        c, _ = pywt.cwt(X[i:i+128], sc, "morl", sampling_period=1/fs, axis=-1, method="fft")
        a = torch.tensor(np.abs(c).transpose(1, 0, 2).astype(np.float32)).unsqueeze(1)          # (b,1,64,T)
        a = F.interpolate(a, size=(side, side), mode="bilinear", align_corners=False)[:, 0]
        mn = a.amin((1, 2), keepdim=True); mx = a.amax((1, 2), keepdim=True)
        out[i:i+128] = ((a - mn)/(mx - mn + 1e-9)).numpy()
    return out

# ------------------------------------------------------------------------------------------------ networks
def build(model, K, T1d):
    import torch, torch.nn as nn
    class Block(nn.Module):                     # pre-norm transformer encoder block
        def __init__(s, d, h, mlp): super().__init__(); s.n1, s.a, s.n2 = nn.LayerNorm(d), nn.MultiheadAttention(d, h, batch_first=True), nn.LayerNorm(d); s.m = nn.Sequential(nn.Linear(d, mlp), nn.GELU(), nn.Linear(mlp, d))
        def forward(s, x): z = s.n1(x); x = x + s.a(z, z, z, need_weights=False)[0]; return x + s.m(s.n2(x))
    class Residual(nn.Module):
        def __init__(s, f): super().__init__(); s.f = f
        def forward(s, x): return s.f(x) + x
    def convmixer(dim, depth, k, p):
        return nn.Sequential(nn.Conv2d(1, dim, p, stride=p), nn.GELU(), nn.BatchNorm2d(dim),
                             *[nn.Sequential(Residual(nn.Sequential(nn.Conv2d(dim, dim, k, groups=dim, padding="same"), nn.GELU(), nn.BatchNorm2d(dim))),
                                             nn.Conv2d(dim, dim, 1), nn.GELU(), nn.BatchNorm2d(dim)) for _ in range(depth)])
    if model == "ibrahim":
        class Net(nn.Module):
            def __init__(s):
                super().__init__(); L, c = [], 1
                for stage in ([64, 64], [128, 128], [256]*4, [512]*4, [512]*5):
                    for o in stage: L += [nn.Conv2d(c, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]; c = o
                    L.append(nn.MaxPool2d(2))
                s.vgg = nn.Sequential(*L); s.emb = nn.Linear(512, 64); s.pos = nn.Parameter(torch.zeros(1, 16, 64)); s.tr = Block(64, 4, 128)
                s.cm = convmixer(32, 7, 5, 5); s.cm_fc = nn.Linear(32, 64); s.alpha = nn.Parameter(torch.ones(1))
                s.res = nn.Linear(64, 64); s.att = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 64), nn.Sigmoid())
                s.drop = nn.Dropout(0.2); s.out = nn.Linear(64, K)
            def forward(s, img, raw):
                x = img.unsqueeze(1); v = s.vgg(x).flatten(2).transpose(1, 2)                 # (B,16,512)
                zc = s.tr(s.emb(v) + s.pos).mean(1)
                zm = s.cm_fc(s.cm(x).mean((2, 3)))
                f = zc + s.alpha*zm; f = torch.relu(f + s.res(f)); f = f*s.att(f)
                return s.out(s.drop(f))
        return Net()
    class Net(nn.Module):                        # rahman: CVT + ConvMixer(+MHSA) + LSTM, concatenated
        def __init__(s):
            super().__init__()
            s.proj = nn.Sequential(nn.Conv2d(1, 64, 7, stride=8, padding=3), nn.GELU(), nn.Conv2d(64, 64, 7, padding=3))
            s.pos = nn.Parameter(torch.zeros(1, 1024, 64)); s.tr = Block(64, 4, 128)
            s.cm = convmixer(32, 7, 5, 8); s.cm_att = nn.MultiheadAttention(32, 4, batch_first=True)
            s.lstm = nn.LSTM(1, 64, batch_first=True); s.out = nn.Linear(160, K)
        def forward(s, img, raw):
            x = img.unsqueeze(1)
            zc = s.tr(s.proj(x).flatten(2).transpose(1, 2) + s.pos).mean(1)
            m = s.cm(x).flatten(2).transpose(1, 2); zm = s.cm_att(m, m, m, need_weights=False)[0].mean(1)
            _, (h, _) = s.lstm(raw.unsqueeze(-1)); zl = h[-1]
            return s.out(torch.cat([zc, zm, zl], 1))
    return Net()

def fit_predict(model, Sg, R, y, tr, va, te, K, seed, threads, smoke):
    import torch, torch.nn.functional as F
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(threads)
    net = build(model, K, R.shape[1]); npar = sum(p.numel() for p in net.parameters())
    epochs = 1 if smoke else (80 if model == "ibrahim" else 60)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=0.01 if model == "ibrahim" else 0.0)
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, 0.95) if model == "ibrahim" else None
    St, Rt, yt = torch.tensor(Sg), torch.tensor(R), torch.tensor(y); rng = np.random.default_rng(seed); hist = []
    trb = tr[:64] if smoke else tr
    for ep in range(epochs):
        net.train(); t0 = time.time(); L = []
        for b in np.array_split(rng.permutation(trb), max(1, len(trb)//32)):
            loss = F.cross_entropy(net(St[b], Rt[b]), yt[b]); opt.zero_grad(); loss.backward(); opt.step(); L.append(loss.item())
        if sch: sch.step()
        hist.append({"epoch": ep, "loss": float(np.mean(L)), "sec": time.time() - t0})
        if ep % 10 == 0 or ep == epochs - 1: print(f"  epoch {ep} loss {hist[-1]['loss']:.3f} ({hist[-1]['sec']:.0f} s)", flush=True)
    net.eval()
    with torch.no_grad():
        P = lambda idx: torch.softmax(torch.cat([net(St[b], Rt[b]) for b in np.array_split(idx, max(1, len(idx)//64))]), 1).numpy()
        return P(va), P(te), int(npar), hist

def chrono(rec, t, frac=0.8):
    tr, te = [], []
    for r in np.unique(rec):
        ii = np.where(rec == r)[0]; ii = ii[np.argsort(t[ii])]; c = int(len(ii)*frac)
        tr += list(ii[:max(1, c - 1)]); te += list(ii[c:])         # drop the boundary segment: no overlap across the cut
    return np.array(tr), np.array(te)

def main():
    from bio_run import eer
    root, model, ds, out = sys.argv[1:5]; th = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5].isdigit() else 8
    smoke = "--smoke" in sys.argv; t00 = time.time()
    sigs, owners, fs, K = load(root, ds)
    X, y, rec, t, fo = segments(sigs, owners, fs, model); Sg = scalograms(X, fo, model)
    print(f"{model} on {ds}: {K} people, {len(y)} segments, scalogram {Sg.shape[1:]}, raw {X.shape[1]} samples", flush=True)
    if model == "ibrahim":                       # 5-fold CV: 20% test, rest 80/20 train/val
        splits = [(a, b) for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)]
    else:                                        # 80/20 per person
        rng = np.random.default_rng(0); a, b = [], []
        for u in np.unique(y): ii = rng.permutation(np.where(y == u)[0]); c = int(0.8*len(ii)); a += list(ii[:c]); b += list(ii[c:])
        splits = [(np.array(a), np.array(b))]
    plans = {"paper_split": splits}
    if ds != "rwppg": plans["chronological_80_20"] = [chrono(rec, t)]   # RW-PPG: independent 6 s recordings, no time order
    res = {"model": model, "dataset": ds, "people": int(K), "segments": int(len(y)), "fs_model": fo, "gaps_filled": GAPS[model]}
    for name, sp in plans.items():
        folds = []
        for k, (tr, te) in enumerate(sp[:1] if smoke else sp):
            r2 = np.random.default_rng(k); va = r2.choice(tr, max(1, len(tr)//5), replace=False); trn = np.setdiff1d(tr, va)
            t0 = time.time(); Pv, Pt, npar, hist = fit_predict(model, Sg, X, y, trn, va, te, K, k, th, smoke)
            L = np.log(np.clip(Pt, 1e-12, 1)); m = np.ones_like(L, bool); m[np.arange(len(te)), y[te]] = False
            folds.append({"top1": float((Pt.argmax(1) == y[te]).mean()), "macro_f1": float(f1_score(y[te], Pt.argmax(1), average="macro")),
                          "val_top1": float((Pv.argmax(1) == y[va]).mean()), "eer": eer(L[np.arange(len(te)), y[te]], L[m]),
                          "params": npar, "minutes": (time.time() - t0)/60, "n_test": int(len(te)), "epochs": len(hist)})
            print(name, "fold", k, {q: round(folds[-1][q], 4) for q in ("top1", "macro_f1", "eer", "minutes")}, flush=True)
        res[name] = {"folds": folds, **{q: float(np.mean([f[q] for f in folds])) for q in ("top1", "macro_f1", "eer")}}
    res["minutes_total"] = (time.time() - t00)/60
    if not smoke: json.dump(res, open(out, "w"), indent=1)
    print("DONE", json.dumps({k: res[k] for k in plans}, default=float)[:400])

if __name__ == "__main__": main()
