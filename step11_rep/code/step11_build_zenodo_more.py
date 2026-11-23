#!/usr/bin/env python3
"""Step 11 Part B data: the teammate's Zenodo biometric cache plus extra TRAINING windows (STEP11_PREDECLARED_PROTOCOL.md).

Replays biometrics/prepare_zenodo.py exactly: same subjects in the same sorted order, the same rng(0) permutation per
(subject, day), the same quality gate (>= 0.35), and the same rules (60 windows per day, try limit 40*60, keep a day with
at least 20, keep a subject with at least 3 days). For every kept day it then continues along the same permutation and
keeps up to 300 accepted windows in total (try limit 40*300), marked is_extra=True. Days and subjects are never added.
The non-extra windows must equal the teammate's cache array-for-array; this is asserted when --check is given.

Usage: python -I step11_build_zenodo_more.py <data_root> [--check <processed/zenodo_bio_50hz_10s.npz>]
Output: <data_root>/processed/step11_zenodo_more_50hz_10s.npz (X, y, sess, vid, users, is_extra)
"""
import os, sys
import numpy as np, h5py
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "capstone_metrics_run", "wavespike-ppg", "biometrics"))
from prepare_zenodo import quality, txt, FS_IN, FS          # the teammate's functions, unchanged

WIN, MAX, MIN_PER_DAY, QMIN = 10, 60, 20, 0.35
MAX_MORE = 300

def main():
    root = sys.argv[1]
    check = sys.argv[sys.argv.index("--check") + 1] if "--check" in sys.argv else None
    raw = f"{root}/raw/zenodo_23597128"; rng = np.random.default_rng(0); n = WIN*FS_IN
    subs = sorted(f[:3] for f in os.listdir(raw) if f.endswith("_PPG.mat"))
    X, Y, S, V, E, users, log = [], [], [], [], [], [], []
    for sid in subs:
        p = h5py.File(f"{raw}/{sid}_PPG.mat"); refs = p["PPG_GREEN"]; by_day = {}
        for k in range(refs.shape[0]):
            day = int(txt(p[p["recording_startday"][k, 0]]))
            L = p[refs[k, 0]].shape[-1]
            by_day.setdefault(day, []).extend((k, i) for i in range(L//n))
        days = {}
        for day, cand in sorted(by_day.items()):
            order = rng.permutation(len(cand)); good, extra = [], []; tried = 0; sig_cache = {}
            pos = 0
            # phase 1: exactly the teammate's loop (break conditions checked before each try, in the same order)
            while pos < len(order):
                if len(good) >= MAX or tried >= 40*MAX: break
                k, i = cand[order[pos]]; pos += 1; tried += 1
                if k not in sig_cache: sig_cache = {k: np.array(p[refs[k, 0]], np.float64).ravel()}
                w = sig_cache[k][i*n:(i+1)*n]
                if quality(w, FS_IN) >= QMIN: good.append(resample_poly(w, FS, FS_IN).astype(np.float32))
            # phase 2 (only for days the teammate keeps): continue along the same permutation
            if len(good) >= MIN_PER_DAY:
                while pos < len(order):
                    if len(good) + len(extra) >= MAX_MORE or tried >= 40*MAX_MORE: break
                    k, i = cand[order[pos]]; pos += 1; tried += 1
                    if k not in sig_cache: sig_cache = {k: np.array(p[refs[k, 0]], np.float64).ravel()}
                    w = sig_cache[k][i*n:(i+1)*n]
                    if quality(w, FS_IN) >= QMIN: extra.append(resample_poly(w, FS, FS_IN).astype(np.float32))
                days[day] = (good, extra)
        log.append(f"subject {sid}: days {sorted(by_day)} -> usable {[(d, len(g), len(x)) for d, (g, x) in sorted(days.items())]}")
        print(log[-1], flush=True)
        if len(days) < 3: continue
        u = len(users); users.append(sid)
        for day, (ws, xs) in days.items():
            for w in ws: X.append(w); Y.append(u); S.append(day); V.append(f"{sid}_d{day}"); E.append(False)
            for w in xs: X.append(w); Y.append(u); S.append(day); V.append(f"{sid}_d{day}"); E.append(True)
    V = np.unique(V, return_inverse=True)[1]
    X, Y, S, E = np.array(X), np.array(Y), np.array(S), np.array(E)
    out = f"{root}/processed/step11_zenodo_more_{FS}hz_{WIN}s.npz"
    np.savez_compressed(out, X=X, y=Y, sess=S, vid=V, users=np.array(users), is_extra=E)
    print(f"saved {out}: {len(X)} windows ({int((~E).sum())} original + {int(E.sum())} extra), {len(users)} subjects")
    if check:
        d = np.load(check, allow_pickle=True); b = ~E
        for key, a in (("X", X[b]), ("y", Y[b]), ("sess", S[b]), ("vid", V[b])):
            assert np.array_equal(d[key], a), f"original windows differ from the teammate cache in {key}"
        assert [str(u) for u in d["users"]] == users, "subject list differs"
        print("CHECK OK: original windows equal the teammate cache array-for-array (X, y, sess, vid, users)")

if __name__ == "__main__": main()
