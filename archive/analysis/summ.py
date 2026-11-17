import glob, json, sys, numpy as np
from sklearn.metrics import roc_auc_score
root, pat, want = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
d = np.load(glob.glob(f"{root}/processed/{pat}_50hz_30s.npz")[0], allow_pickle=True); y, g = d["y"], d["subject"].astype(str)
R = {}
for p in glob.glob(f"{root}/results/{pat}_v3_*/*_cv*_ts*_fold*.json"):
    r = json.load(open(p))
    if r["variant"] in want: R.setdefault((r["variant"], r["cv_seed"], r["train_seed"]), []).append(r)
print(f"{'variant':14s} {'rep':6s} {'winAUROC b/l':>14s} {'macroF1 b/l':>13s} {'withinSubjAUROC b/l':>20s}")
for (v, cs, ts), rr in sorted(R.items()):
    if len(rr) < 5: continue
    rr = sorted(rr, key=lambda r: r["fold"]); idx = np.concatenate([r["te_idx"] for r in rr]); out = []
    for ck in ("best", "last"):
        P = np.concatenate([np.array(r[ck]["prob"]) for r in rr]); p = P[:, 1] / P.sum(1)
        from sklearn.metrics import f1_score
        ws = [roc_auc_score(y[idx][g[idx] == s], p[g[idx] == s]) for s in sorted(set(g[idx])) if len(set(y[idx][g[idx] == s])) == 2]
        out.append((roc_auc_score(y[idx], p), f1_score(y[idx], P.argmax(1), average="macro"), np.mean(ws) if ws else float("nan")))
    print(f"{v:14s} cv{cs}ts{ts} {out[0][0]:6.3f}/{out[1][0]:.3f} {out[0][1]:6.3f}/{out[1][1]:.3f} {out[0][2]:9.3f}/{out[1][2]:.3f}")
