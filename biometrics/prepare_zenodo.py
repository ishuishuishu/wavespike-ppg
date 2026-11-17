#!/usr/bin/env python3
"""Zenodo 23597128 wrist PPG -> identity windows with the recording DAY as the session (cross-day biometrics).
Per (subject, day): up to MAX_PER_DAY random 10 s windows that pass a quality gate (finite, not flat, a clear
heart-rate peak in the spectrum). PPG_GREEN 100 Hz -> 50 Hz. Days with < MIN_PER_DAY good windows are dropped;
subjects with < 3 usable days are dropped (the session folds need test + val + train days).
Usage: python -I biometrics/prepare_zenodo.py <data_root> [win_s=10] [max_per_day=60]
"""
import os, sys
import numpy as np, h5py
from scipy.signal import resample_poly

FS_IN, FS = 100, 50
def txt(ds): return "".join(chr(int(c)) for c in np.array(ds).ravel())

def quality(w, fs):
    if not np.all(np.isfinite(w)) or w.std() < 1e-6: return 0.0
    if np.max(np.bincount(np.unique(w, return_inverse=True)[1])) > 0.2*len(w): return 0.0    # clipped / stuck values
    x = w - np.convolve(w, np.ones(fs)/fs, "same"); x = x[fs:-fs]
    X = np.abs(np.fft.rfft(x*np.hanning(len(x))))**2; f = np.fft.rfftfreq(len(x), 1/fs)
    b = (f > 0.7) & (f < 3.5); i = np.argmax(X*b)
    return float(X[b & (np.abs(f - f[i]) < 0.15)].sum()/X[(f > 0.5) & (f < 8)].sum())

def main():
    root = sys.argv[1]; win = int(sys.argv[2]) if len(sys.argv) > 2 else 10; MAX = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    MIN_PER_DAY, QMIN = 20, 0.35
    raw = f"{root}/raw/zenodo_23597128"; rng = np.random.default_rng(0); n = win*FS_IN
    subs = sorted(f[:3] for f in os.listdir(raw) if f.endswith("_PPG.mat"))
    X, Y, S, V, users = [], [], [], [], []
    for sid in subs:
        p = h5py.File(f"{raw}/{sid}_PPG.mat"); refs = p["PPG_GREEN"]; by_day = {}
        for k in range(refs.shape[0]):
            day = int(txt(p[p["recording_startday"][k, 0]]))
            L = p[refs[k, 0]].shape[-1]
            by_day.setdefault(day, []).extend((k, i) for i in range(L//n))
        days = {}
        for day, cand in sorted(by_day.items()):
            order = rng.permutation(len(cand)); good = []; tried = 0
            sig_cache = {}
            for j in order:
                if len(good) >= MAX or tried >= 40*MAX: break
                k, i = cand[j]; tried += 1
                if k not in sig_cache: sig_cache = {k: np.array(p[refs[k, 0]], np.float64).ravel()}
                w = sig_cache[k][i*n:(i+1)*n]
                if quality(w, FS_IN) >= QMIN: good.append(resample_poly(w, FS, FS_IN).astype(np.float32))
            if len(good) >= MIN_PER_DAY: days[day] = good
        print(f"subject {sid}: days {sorted(by_day)} -> usable {[(d, len(v)) for d, v in sorted(days.items())]}", flush=True)
        if len(days) < 3: continue
        u = len(users); users.append(sid)
        for day, ws in days.items():
            for w in ws: X.append(w); Y.append(u); S.append(day); V.append(f"{sid}_d{day}")
    V = np.unique(V, return_inverse=True)[1]                       # "recording" = (subject, day) for decision aggregation
    out = f"{root}/processed/zenodo_bio_{FS}hz_{win}s.npz"
    np.savez_compressed(out, X=np.array(X), y=np.array(Y), sess=np.array(S), vid=V, users=np.array(users))
    print(f"saved {out}: {len(X)} windows, {len(users)} subjects")

if __name__ == "__main__": main()
