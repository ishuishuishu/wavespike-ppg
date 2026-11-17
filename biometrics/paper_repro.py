#!/usr/bin/env python3
"""Re-run recreated published methods under each paper's OWN protocol, on open data, to check that our recreations
reach the accuracy the papers report when the conditions match.

  lovisotto  Seeing Red (Lovisotto et al. 2020): per-beat features + RBF SVM, random stratified 2-fold over beats
             (same videos in train and test), decisions from n randomly drawn beats (n = 1, 2, 5, 10, 20).
             Paper: single-beat EER about 15%, < 5% above 10 beats, SVM < 1% at 20 beats.
  yadav      Yadav et al. 2017 (CWT + DLDA): train on the first 45 s of each recording, test on the rest (same session),
             decisions from nTest consecutive segments. Paper: CapnoBase EER 0.46% (all test data), 1.1% (2 segments).
             Run on BIDMC and MIMIC PERform (CapnoBase's licence is unclear). DLDA ~ PCA + shrinkage LDA.
  cvt        CVT-ConvMixer-style CWT scalogram + CNN: 5 s segments, 50% overlap, random stratified split.
             Paper: 95% accuracy (BIDMC + MIMIC + RW-PPG). Also a chronological split (no overlap leakage).
  lowfr      Low-frame-rate PPG paper protocol on Seeing Red (their CFISHR data is not open): 5 s segments, 50%
             overlap, random 80/20 split per person. Paper: 97.7-98% accuracy on CFISHR.
Usage: python -I biometrics/paper_repro.py <data_root> <protocol> <out.json> [threads]
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bio_run import eer, cwt_cnn_probs, bidmc_signals, CHANNELS
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score

FS = 50
def bp(x, lo, hi, fs=FS): return sosfiltfilt(butter(4, [lo, hi], "bandpass", fs=fs, output="sos"), x)
def lp(x, hi, fs=FS): return sosfiltfilt(butter(4, hi, "lowpass", fs=fs, output="sos"), x)

def seeing_red_luma(root):
    d = np.load(f"{root}/raw/seeing_red/sr_rgb50.npz", allow_pickle=True); w = np.array(CHANNELS["luma"])
    users = d["user"].astype(str); U = sorted(set(users)); uid = {u: i for i, u in enumerate(U)}
    return [np.asarray(r, float) @ w for r in d["ppg"]], np.array([uid[u] for u in users]), U

def mimic_signals(root):
    d = np.load(f"{root}/processed/mimic_perform_af_50hz_30s.npz", allow_pickle=True); s = d["subject"].astype(str)
    U = sorted(set(s)); return [np.concatenate(list(d["X"][s == u])) for u in U], U

def trials_eer(S, y):
    """S: (decisions, K) scores, y: true identity -> pooled EER (genuine = own column, impostor = all other columns)."""
    m = np.ones_like(S, bool); m[np.arange(len(y)), y] = False
    return eer(S[np.arange(len(y)), y], S[m])

# ----------------------------------------------------------------------------- Lovisotto et al. 2020 (Seeing Red)
def beat_feats(lvl, fs=FS):
    x = -lvl[fs:]; x = x - np.convolve(x, np.ones(fs)/fs, "same"); x = lp(x, 4.0)       # 1 s detrend + 4 Hz low-pass (paper)
    pk, _ = find_peaks(x, distance=int(0.33*fs), prominence=0.3*x.std()); pre, post = int(0.35*fs), int(0.55*fs)
    F = []
    for j in range(1, len(pk) - 1):
        p = pk[j]
        if p - pre < 0 or p + post > len(x): continue
        b = x[p-pre:p+post]; t = np.interp(np.linspace(0, len(b) - 1, 60), np.arange(len(b)), b)
        z = (t - t.mean())/(t.std() + 1e-9); dz = np.diff(z)
        ff = np.abs(np.fft.rfft(z))[1:11]
        F.append(np.concatenate([z, dz, ff, [b.max() - b.min(), lvl[fs:][p-pre:p+post].mean(), (pk[j]-pk[j-1])/fs,
                                             (pk[j+1]-pk[j])/fs, skew(b), kurtosis(b), np.argmax(t)/60, np.argmin(np.gradient(t))/60]]))
    return np.asarray(F)

def lovisotto(root, rng=np.random.default_rng(0)):
    sigs, vid_user, U = seeing_red_luma(root); F, y = [], []
    for s, u in zip(sigs, vid_user):
        f = beat_feats(s)
        if len(f): F.append(f); y += [u]*len(f)
    F = np.concatenate(F); y = np.array(y); K = len(U); res = {"beats": int(len(y)), "people": K, "folds": []}
    for k, (a, b) in enumerate(StratifiedKFold(2, shuffle=True, random_state=0).split(F, y)):
        clf = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True, random_state=0)).fit(F[a], y[a])
        L = np.log(np.clip(clf.predict_proba(F[b]), 1e-12, 1)); yt = y[b]; fold = {}
        for n in (1, 2, 5, 10, 20):
            G, I, acc = [], [], []
            for u in range(K):
                iu = np.where(yt == u)[0]
                for _ in range(50):                                   # genuine sequences of n beats of user u
                    s = L[rng.choice(iu, n, replace=len(iu) < n)].mean(0); G.append(s[u]); acc.append(int(np.argmax(s) == u))
                for v in range(K):                                    # 10 impostor sequences from every other user
                    if v == u: continue
                    iv = np.where(yt == v)[0]
                    for _ in range(10): I.append(L[rng.choice(iv, n, replace=len(iv) < n)].mean(0)[u])
            fold[n] = {"eer": eer(np.array(G), np.array(I)), "top1": float(np.mean(acc))}
        res["folds"].append(fold)
    res["by_n"] = {n: {m: float(np.mean([f[n][m] for f in res["folds"]])) for m in ("eer", "top1")} for n in (1, 2, 5, 10, 20)}
    return res

# ----------------------------------------------------------------------------- Yadav et al. 2017 (CWT + DLDA)
def cwt_seg_feats(seg, fs=FS):
    import pywt
    fr = np.geomspace(0.5, 8, 24); c, _ = pywt.cwt(seg, pywt.central_frequency("morl")*fs/fr, "morl", sampling_period=1/fs)
    a = np.abs(c)[:, ::5]; return ((a - a.mean())/(a.std() + 1e-9)).ravel()

def yadav(root, which):
    sigs, U = (bidmc_signals(root) if which == "bidmc" else mimic_signals(root)); seg = 3*FS; ntr = 45*FS // seg
    Xtr, ytr, Xte, yte, pos = [], [], [], [], []
    for i, s in enumerate(sigs):
        x = bp(np.asarray(s, float), 0.5, 8); segs = [x[a:a+seg] for a in range(0, len(x) - seg + 1, seg)]
        for j, sg in enumerate(segs):
            f = cwt_seg_feats(sg)
            if j < ntr: Xtr.append(f); ytr.append(i)
            else: Xte.append(f); yte.append(i); pos.append(j)
    Xtr, Xte, ytr, yte = map(np.asarray, (Xtr, Xte, ytr, yte))
    clf = make_pipeline(StandardScaler(), PCA(n_components=min(100, len(ytr) - len(U)), random_state=0),
                        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")).fit(Xtr, ytr)
    L = np.log(np.clip(clf.predict_proba(Xte), 1e-12, 1)); out = {"people": len(U), "train_s_per_person": 45, "segment_s": 3,
                                                                  "test_segments": int(len(yte)), "by_n": {}}
    for n in (1, 2, 5, 10, 20, "all"):
        S, yy = [], []
        for i in range(len(U)):
            ii = np.where(yte == i)[0]; m = len(ii) if n == "all" else n
            for a in range(0, len(ii) - m + 1, m): S.append(L[ii[a:a+m]].mean(0)); yy.append(i)
        S, yy = np.array(S), np.array(yy)
        out["by_n"][str(n)] = {"eer": trials_eer(S, yy), "top1": float((S.argmax(1) == yy).mean()), "decisions": int(len(yy))}
    return out

# ----------------------------------------------------------------------------- CWT scalogram + CNN protocols
def segments(sigs, owners, win_s=5, hop_s=2.5, hp=True):
    X, y, rec, t = [], [], [], []
    for r, (s, o) in enumerate(zip(sigs, owners)):
        x = bp(np.asarray(s, float), 0.5, 8) if hp else np.asarray(s, float); n, h = int(win_s*FS), int(hop_s*FS)
        for a in range(0, len(x) - n + 1, h): w = x[a:a+n]; X.append((w - w.mean())/(w.std() + 1e-9)); y.append(o); rec.append(r); t.append(a)
    return np.asarray(X, np.float32), np.asarray(y), np.asarray(rec), np.asarray(t)

def cnn_eval(X, y, splits, K, threads, tag):
    A = np.zeros((len(y), 2), np.float32); folds = []
    for k, (tr, te) in enumerate(splits):
        t0 = time.time(); rng = np.random.default_rng(k); va = rng.choice(tr, max(1, len(tr)//5), replace=False)
        trn = np.setdiff1d(tr, va); Pv, Pt, npar = cwt_cnn_probs(X, A, y, trn, va, te, K, k, threads, FS, epochs=60)
        folds.append({"top1": float((Pt.argmax(1) == y[te]).mean()), "macro_f1": float(f1_score(y[te], Pt.argmax(1), average="macro")),
                      "eer": trials_eer(np.log(np.clip(Pt, 1e-12, 1)), y[te]), "minutes": (time.time() - t0)/60, "params": int(npar)})
        print(tag, "fold", k, {m: round(folds[-1][m], 4) for m in ("top1", "macro_f1", "eer")}, flush=True)
    return {"folds": folds, **{m: float(np.mean([f[m] for f in folds])) for m in ("top1", "macro_f1", "eer")}}

def chrono_splits(rec, t, frac=0.8):
    tr, te = [], []
    for r in np.unique(rec):
        ii = np.where(rec == r)[0]; ii = ii[np.argsort(t[ii])]; c = int(len(ii)*frac)
        tr += list(ii[:c - 2]); te += list(ii[c:])                 # drop 2 boundary segments: no overlap across the cut
    return [(np.array(tr), np.array(te))]

def cvt(root, which, threads):
    sigs, U = (bidmc_signals(root) if which == "bidmc" else mimic_signals(root))
    X, y, rec, t = segments(sigs, range(len(sigs)))
    rnd = [(a, b) for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)]
    return {"people": len(U), "segments": int(len(y)), "random_5fold": cnn_eval(X, y, rnd, len(U), threads, f"{which} random"),
            "chronological_80_20": cnn_eval(X, y, chrono_splits(rec, t), len(U), threads, f"{which} chrono")}

def lowfr(root, threads):
    sigs, owners, U = seeing_red_luma(root)
    X, y, rec, t = segments([-s[FS:] for s in sigs], owners)
    tr, te = [], []
    rng = np.random.default_rng(0)
    for u in np.unique(y):
        ii = rng.permutation(np.where(y == u)[0]); c = int(0.8*len(ii)); tr += list(ii[:c]); te += list(ii[c:])
    return {"people": len(U), "segments": int(len(y)),
            "random_80_20_per_person": cnn_eval(X, y, [(np.array(tr), np.array(te))], len(U), threads, "seeing_red random"),
            "chronological_80_20": cnn_eval(X, y, chrono_splits(rec, t), len(U), threads, "seeing_red chrono")}

if __name__ == "__main__":
    root, proto, out = sys.argv[1], sys.argv[2], sys.argv[3]; th = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    t0 = time.time()
    r = {"lovisotto": lambda: lovisotto(root), "yadav_bidmc": lambda: yadav(root, "bidmc"), "yadav_mimic": lambda: yadav(root, "mimic"),
         "cvt_bidmc": lambda: cvt(root, "bidmc", th), "cvt_mimic": lambda: cvt(root, "mimic", th), "lowfr": lambda: lowfr(root, th)}[proto]()
    r["minutes_total"] = (time.time() - t0)/60
    json.dump(r, open(out, "w"), indent=1, default=float); print(proto, json.dumps(r.get("by_n", {k: r[k] for k in r if isinstance(r[k], (int, float))}), default=float)[:600])
