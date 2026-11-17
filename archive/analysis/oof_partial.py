import glob, json, sys, numpy as np
from sklearn.metrics import roc_auc_score, f1_score
root = sys.argv[1]
d = np.load(f"{root}/processed/mimic_perform_af_50hz_30s.npz", allow_pickle=True)
y, g = d["y"], d["subject"].astype(str)
R = {}
for p in glob.glob(f"{root}/results/mimic_perform_af_v3_*/*_cv*_ts*_fold*.json"):
    r = json.load(open(p)); R.setdefault((r["variant"], r["cv_seed"], r["train_seed"]), []).append(r)
print("complete repeats (variant, cv, ts) with 5 folds:", sum(len(v) == 5 for v in R.values()), "| partial:", sum(len(v) < 5 for v in R.values()))
for ck in ("best", "last"):
    print(f"\n== checkpoint: {ck}  (cv0, ts0) pooled out-of-fold ==")
    print(f"{'variant':14s} {'winAUROC':>9s} {'macroF1':>8s} {'subjAUROC':>10s} {'epochs':>7s}  per-fold AUROC")
    for (v, cs, ts), rr in sorted(R.items()):
        if len(rr) < 5: continue
        rr = sorted(rr, key=lambda r: r["fold"]); idx = np.concatenate([r["te_idx"] for r in rr])
        P = np.concatenate([np.array(r[ck]["prob"]) for r in rr]); P = P / P.sum(1, keepdims=True)
        assert len(set(idx)) == len(y)
        yt, gt = y[idx], g[idx]; p = P[:, 1]
        subs = sorted(set(gt)); sp = np.array([p[gt == s].mean() for s in subs]); sl = np.array([int((yt[gt == s] == 1).any()) for s in subs])
        print(f"{v:14s} {roc_auc_score(yt, p):9.3f} {f1_score(yt, P.argmax(1), average='macro'):8.3f} {roc_auc_score(sl, sp):10.3f} "
              f"{np.mean([r['epochs_run'] for r in rr]):7.1f}  " + " ".join(f"{r[ck]['auroc']:.2f}" for r in rr) + f"   (cv{cs} ts{ts}, {sum(r['minutes'] for r in rr):.0f} min)")
