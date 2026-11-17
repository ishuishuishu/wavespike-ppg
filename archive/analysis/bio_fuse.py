#!/usr/bin/env python3
"""Offline score fusion / ensembling of saved bio_run results (no training).
Members' log-probabilities are combined as  w*mean(log P_neural) + (1-w)*log P_classical ; w is chosen per fold on
VALIDATION EER (n windows per decision), then the test metrics are reported at that w. w=1 -> neural only, w=0 -> classical.
Usage: python -I biometrics/bio_fuse.py <res_dir> <classical|-> <neural1[+neural2...]> [ckpt=last] [seeds=0,1]
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "biometrics"))   # finds bio_run.py
from bio_run import bio_metrics as _bm, aggregate, trials, eer, tau_at_far

def bio_metrics(Pv, yv, vv, Pt, yt, vt, n, far_target=0.02):
    """ZNORM=1: per-claimed-identity z-norm of the decision log-probability, mean/std taken from that identity's VALIDATION
    impostor scores (no test information). Identification (top1/F1) is unchanged by construction (it uses the raw argmax)."""
    m = _bm(Pv, yv, vv, Pt, yt, vt, n, far_target)
    if os.environ.get("ZNORM") != "1": return m
    Av, yv2 = aggregate(Pv, yv, vv, n); At, yt2 = aggregate(Pt, yt, vt, n)
    Lv, Lt = np.log(np.clip(Av, 1e-12, 1)), np.log(np.clip(At, 1e-12, 1)); K = Lv.shape[1]
    imp = np.ones_like(Lv, bool); imp[np.arange(len(yv2)), yv2] = False
    mu = np.array([Lv[imp[:, c], c].mean() for c in range(K)]); sd = np.array([Lv[imp[:, c], c].std() + 1e-6 for c in range(K)])
    Zv, Zt = (Lv - mu)/sd, (Lt - mu)/sd
    gv, iv_, _ = trials(Zv, yv2); gt, it, _ = trials(Zt, yt2); tau = tau_at_far(iv_, far_target)
    m.update(eer=eer(gt, it), eer_val=eer(gv, iv_), tau=tau, far=float((it >= tau).mean()), frr=float((gt < tau).mean()))
    return m

def load(res, v, k, s, ck):
    p = f"{res}/{v}_f{k}_ts{s}.json"
    if not os.path.exists(p): return None
    r = json.load(open(p)); return r, np.array(r[ck]["p_val"]), np.array(r[ck]["p_te"])

def main():
    res, cl, neu = sys.argv[1], sys.argv[2], sys.argv[3].split("+")
    ck = sys.argv[4] if len(sys.argv) > 4 else "last"; seeds = [int(s) for s in (sys.argv[5] if len(sys.argv) > 5 else "0,1").split(",")]
    W = np.round(np.linspace(0, 1, 11), 2) if cl != "-" else [1.0]
    print(f"classical={cl} neural={neu} ckpt={ck} seeds={seeds}")
    for n in (1, 3, 5):
        out = []
        for k in range(3):
            mem = [m for v in neu for s in seeds if (m := load(res, v, k, s, ck)) is not None]
            if not mem: continue
            r0 = mem[0][0]; yv, vv, yt, vt = (np.array(r0[x]) for x in ("y_val", "vid_val", "y_te", "vid_te"))
            assert all(m[0]["y_te"] == r0["y_te"] and m[0]["y_val"] == r0["y_val"] for m in mem)
            lnv = np.mean([np.log(np.clip(m[1], 1e-12, 1)) for m in mem], 0); lnt = np.mean([np.log(np.clip(m[2], 1e-12, 1)) for m in mem], 0)
            if cl != "-":
                c = load(res, cl, k, 0, "last"); assert c[0]["y_te"] == r0["y_te"]
                lcv, lct = np.log(np.clip(c[1], 1e-12, 1)), np.log(np.clip(c[2], 1e-12, 1))
            else: lcv, lct = 0*lnv, 0*lnt
            def P(a): e = np.exp(a - a.max(1, keepdims=True)); return e/e.sum(1, keepdims=True)
            best = None
            for w in W:
                m = bio_metrics(P(w*lnv + (1-w)*lcv), yv, vv, P(w*lnt + (1-w)*lct), yt, vt, n)
                if best is None or m["eer_val"] < best[1]["eer_val"] - 1e-12: best = (w, m)
            out.append((k, len(mem), *best))
        if not out: print("no runs"); return
        f = lambda key: np.mean([o[3][key] for o in out])
        print(f"n={n}: " + " | ".join(f"f{k} m{nm} w={w:.1f} valEER {100*m['eer_val']:.1f} EER {100*m['eer']:.1f} top1 {m['top1']:.3f}"
                                     for k, nm, w, m in out)
              + f"\n      MEAN valEER {100*f('eer_val'):.1f} | test EER {100*f('eer'):.1f} | top1 {f('top1'):.3f} | F1 {f('macro_f1'):.3f}"
              f" | FAR {100*f('far'):.1f} FRR {100*f('frr'):.1f}")

if __name__ == "__main__": main()
