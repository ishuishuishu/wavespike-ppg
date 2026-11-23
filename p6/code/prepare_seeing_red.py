#!/usr/bin/env python3
"""Seeing Red (Oxford, ODC-PDDL): turn each finger-on-camera video into a PPG trace.
PPG = mean red-channel intensity per frame (whole frame), inverted so systole is up, resampled to 50 Hz.
Output: <out>.npz with per-video arrays: ppg (object), user, ts (relative seconds), fps.
Usage: python -I biometrics/prepare_seeing_red.py <videos_dir> <data_root>/raw/seeing_red/sr_rgb50.npz [workers]
"""
import glob, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

FS_OUT = 50

def one(path):
    import cv2
    cv2.setNumThreads(1)
    v = cv2.VideoCapture(path); fps = v.get(cv2.CAP_PROP_FPS); r = []
    while True:
        ok, fr = v.read()
        if not ok: break
        r.append(fr.reshape(-1, 3).mean(0)[::-1])   # OpenCV is BGR -> store R, G, B frame means
    r = np.asarray(r, np.float64)                   # (frames, 3), raw intensities 0..255 (NOT inverted)
    t = np.arange(len(r)) / fps
    tn = np.arange(0, t[-1], 1 / FS_OUT)
    k = max(int(round(fps / FS_OUT)), 1)            # anti-alias: moving mean over one output bin, then interpolate (240 -> 50 Hz)
    x = np.stack([np.interp(tn, t, np.convolve(r[:, c], np.ones(k) / k, mode="same")) for c in range(3)], 1).astype(np.float32)
    user = os.path.basename(os.path.dirname(path)); ts = int(os.path.basename(path)[:-4])
    return user, ts, fps, x

if __name__ == "__main__":
    vids = sorted(glob.glob(os.path.join(sys.argv[1], "*", "*.mp4")))
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    with ProcessPoolExecutor(w) as ex: res = list(ex.map(one, vids))
    users = np.array([r[0] for r in res]); ts = np.array([r[1] for r in res]); fps = np.array([r[2] for r in res])
    ppg = np.empty(len(res), object); ppg[:] = [r[3] for r in res]
    np.savez(sys.argv[2], ppg=ppg, user=users, ts=ts, fps=fps, fs=FS_OUT)
    L = np.array([len(p) / FS_OUT for p in ppg])
    print(f"{len(res)} videos, {len(set(users))} users, duration s min {L.min():.1f} median {np.median(L):.1f} max {L.max():.1f}, fps {sorted(set(np.round(fps,1)))}")
