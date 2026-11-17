#!/usr/bin/env python3
"""Every metric stored in every result JSON, aggregated (mean, SD, min, max over runs) -> one JSON for the results page."""
import glob, json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

root = sys.argv[1]; R = f"{root}/results"; out = {"bio": {}, "af": {}, "cinc": None}
S = lambda a: {"mean": round(float(np.nanmean(a)), 4), "sd": round(float(np.nanstd(a)), 4),
               "min": round(float(np.nanmin(a)), 4), "max": round(float(np.nanmax(a)), 4)} if np.isfinite(a).any() else None
BKEYS = ["top1", "macro_f1", "eer", "eer_val", "auc", "far", "frr", "hter", "tau", "n_decisions", "n_genuine", "n_impostor"]
for d in sorted(glob.glob(f"{R}/bio_*")):
    ds = os.path.basename(d); vs = sorted({os.path.basename(p).rsplit("_f", 1)[0] for p in glob.glob(d + "/*.json")})
    if not vs: continue
    out["bio"][ds] = {}
    for v in vs:
        rs = [json.load(open(p)) for p in sorted(glob.glob(f"{d}/{v}_f*_ts*.json"))]
        e = {"runs": len(rs), "params": rs[0].get("params"),
             "extra": {k: S(np.array([r.get(k, np.nan) for r in rs], float)) for k in ("epochs_run", "best_epoch", "hid_rate_last", "minutes")}}
        for ck in ("best", "last"):
            e[ck] = []
            for j in range(3):
                M = [r[ck]["metrics"][j] for r in rs]
                m = {k: S(np.array([x.get(k, np.nan) for x in M], float)) for k in BKEYS}
                ids = sorted({i for x in M for i in x.get("frr_per_id", {})}, key=int)
                m["frr_per_id"] = {i: round(float(np.mean([x["frr_per_id"][i] for x in M if i in x.get("frr_per_id", {})])), 4) for i in ids}
                m["n_agg"] = M[0].get("n_agg"); e[ck].append(m)
        out["bio"][ds][v] = e
for ds, rd in [("mimic_perform_af", "mimic_perform_af_v3_219af732"), ("zenodo_wrist_af", "zenodo_wrist_af_v3_3f4c36c5")]:
    z = np.load(f"{root}/processed/{ds}_50hz_30s.npz", allow_pickle=True); y = z["y"]; g = z["subject"].astype(str)
    Rr = {}
    for p in glob.glob(f"{R}/{rd}/*_cv*_ts*_fold*.json"):
        r = json.load(open(p)); Rr.setdefault(r["variant"], {}).setdefault((r["cv_seed"], r["train_seed"]), []).append(r)
    out["af"][ds] = {}
    for v in sorted(Rr):
        reps = [rr for rr in Rr[v].values() if len(rr) == 5]
        if not reps: continue
        e = {"repeats": len(reps)}
        for ck in ("best", "last"):
            rows = []
            for rr in reps:
                idx = np.concatenate([r["te_idx"] for r in rr]); P = np.concatenate([np.array(r[ck]["prob"]) for r in rr])
                p = P[:, 1]/P.sum(1); pred = P.argmax(1); yt = y[idx]; tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
                df = pd.DataFrame({"g": g[idx], "p": p, "a": yt}).groupby("g").agg(p=("p", "mean"), a=("a", "max"))
                ws = [roc_auc_score(yt[g[idx] == s], p[g[idx] == s]) for s in sorted(set(g[idx])) if len(set(yt[g[idx] == s])) == 2]
                fa = [r[ck]["auroc"] for r in rr]
                rows.append({"auroc": roc_auc_score(yt, p), "macro_f1": f1_score(yt, pred, average="macro"), "acc": (pred == yt).mean(),
                             "sens": tp/max(tp+fn, 1), "spec": tn/max(tn+fp, 1), "ppv": tp/max(tp+fp, 1), "npv": tn/max(tn+fn, 1),
                             "subj_auroc": roc_auc_score(df.a, df.p) if df.a.nunique() == 2 else np.nan,
                             "within_subj_auroc": np.mean(ws) if ws else np.nan,
                             "fold_auroc_mean": np.nanmean(fa), "fold_auroc_sd": np.nanstd(fa),
                             "epochs": np.mean([r.get("epochs_run", np.nan) for r in rr]), "best_epoch": np.mean([r.get("best_epoch", np.nan) for r in rr])})
            e[ck] = {k: S(np.array([x[k] for x in rows], float)) for k in rows[0]}
        out["af"][ds][v] = e
c = f"{R}/cinc2015_v3/oof.json"
if os.path.exists(c): out["cinc"] = json.load(open(c))
json.dump(out, open(sys.argv[2], "w"), separators=(",", ":"))
print({k: len(v) for k, v in out["bio"].items()}, {k: len(v) for k, v in out["af"].items()}, os.path.getsize(sys.argv[2]))
