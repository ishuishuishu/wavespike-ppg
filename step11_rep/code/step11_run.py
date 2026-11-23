#!/usr/bin/env python3
"""Step 11 Part B runner (STEP11_PREDECLARED_PROTOCOL.md): the teammate's bio_run.py main loop, unchanged in every
respect except (a) the loader reads step11_zenodo_more_50hz_10s.npz and (b) extra windows are kept out of validation
and test through bio_run.session_folds(..., E=~is_extra), the same mask mechanism bio_run uses for Seeing Red.

--base runs the same code on the teammate's original cache (no extra windows); used once to check that this runner
reproduces bio_run.py exactly (Part A beat_svm_amp probabilities).

Usage: python -I step11_run.py --variants x_tau_amp_more --folds 0,1,2 --seeds 0,1 --threads 4 [--base]
Variant names: <teammate variant>_more (e.g. x_tau_amp_more, cwt_cnn_amp_more, x_h128_tau_amp_more) or, with --base,
the teammate variant name itself.
Results: $WAVESPIKE_DATA_ROOT/results/step11_more_w10[ _basecheck]/<variant>_f<fold>_ts<seed>.json (bio_run format).
"""
import argparse, json, os, sys, time
import numpy as np
REPO = os.path.join(os.environ.get("WAVESPIKE_REPO", os.path.join(os.path.expanduser("~"), "capstone_metrics_run", "wavespike-ppg")), "biometrics")
sys.path.insert(0, REPO)
import bio_run as B

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True); ap.add_argument("--folds", default="0,1,2"); ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--base", action="store_true")
    a = ap.parse_args()
    root = os.environ["WAVESPIKE_DATA_ROOT"]
    import torch; torch.set_num_threads(a.threads)
    g = B.load_notebook_code(); g["VARIANTS"].update(B.BIO_VARIANTS); B.make_amp_fusion(g); B.make_packet(g)
    if a.base:
        d = np.load(f"{root}/processed/zenodo_bio_50hz_10s.npz", allow_pickle=True); E = np.ones(len(d["y"]), bool)
        res_dir = f"{root}/results/step11_more_w10_basecheck"
    else:
        d = np.load(f"{root}/processed/step11_zenodo_more_50hz_10s.npz", allow_pickle=True); E = ~d["is_extra"]
        res_dir = f"{root}/results/step11_more_w10"
    os.makedirs(res_dir, exist_ok=True)
    X, y, S, V, U = d["X"].astype(np.float32), d["y"], d["sess"], d["vid"], [str(u) for u in d["users"]]
    AMP = np.stack([X.mean(1), np.log(X.std(1) + 1e-6)], 1)                        # as bio_run.main (no loader-side level)
    X = ((X - X.mean(1, keepdims=True))/(X.std(1, keepdims=True) + 1e-6)).astype(np.float32)
    CFG = g["CFG"]; CFG.n_classes = len(U); CFG.win_sec = 10
    folds = B.session_folds(y, S, E=None if a.base else E)
    print(f"step11 {'BASE' if a.base else 'MORE'}: X {X.shape} | identities {len(U)} | extra windows {int((~E).sum())}")
    for k, (tr, va, te) in enumerate(folds):
        assert E[va].all() and E[te].all(), "an extra window leaked into validation or test"
        print(f"fold {k}: train {len(tr)} val {len(va)} test {len(te)} windows")
    for vname in a.variants.split(","):
        base_v = vname[:-5] if vname.endswith("_more") else vname
        assert a.base == (base_v == vname), "use <variant>_more without --base, the plain name with --base"
        spec = g["VARIANTS"][base_v]
        for k in map(int, a.folds.split(",")):
            for ts in map(int, a.seeds.split(",")):
                p = f"{res_dir}/{vname}_f{k}_ts{ts}.json"
                if os.path.exists(p): print("skip (done)", p); continue
                if spec.get("_kind") == "beat_lr" and ts != int(a.seeds.split(",")[0]): continue
                tr, va, te = folds[k]; t0 = time.time(); extra = {}
                # ---- below: bio_run.main's per-run body, verbatim apart from the variable names g/AMP/X/y/V/U ----
                if spec.get("_kind") == "beat_lr":
                    from sklearn.linear_model import LogisticRegression; from sklearn.preprocessing import StandardScaler
                    from sklearn.pipeline import make_pipeline
                    fk = "FEATS_" + spec.get("_feat", "beat")
                    if fk not in g: g[fk] = B.wavelet_features(X, CFG.fs) if spec.get("_feat") == "dwt" else B.beat_features(X, CFG.fs)
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
                    Pv, Pt, npar = B.cwt_cnn_probs(X, AMP, y, tr, va, te, len(U), ts, a.threads, CFG.fs,
                                                   epochs=60, use_amp=spec.get("_amp", False))
                    ent = {"best": (Pv, Pt), "last": (Pv, Pt)}; extra = {"params": int(npar)}
                else:
                    c = g["variant_config"](base_v, ts, base=CFG); c.n_classes = len(U)
                    Xa = X
                    if getattr(c, "amp_fusion", False):
                        mu, sd = AMP[tr].mean(0), AMP[tr].std(0) + 1e-6
                        Xa = np.concatenate([X, ((AMP - mu)/sd).astype(np.float32)], 1); g["AMP_COLS"] = AMP.shape[1]
                        extra["amp_mu"], extra["amp_sd"] = mu.tolist(), sd.tolist()        # needed by the Pi export
                    else: g["AMP_COLS"] = 0
                    out = g["train"](c, tr, va, verbose=False, Xa=Xa, ya=y)
                    ent = {nm: (g["predict"](out[nm], va, Xa=Xa), g["predict"](out[nm], te, Xa=Xa)) for nm in ("best", "last")}
                    extra.update({"epochs_run": len(out["hist"]), "best_epoch": out["best_epoch"],
                                  "hid_rate_last": out["hist"][-1]["hid_rate"], "params": int(g["n_params"](out["best"]))})
                    import torch as _t
                    _t.save(out["last_state"], p[:-5] + "_last.pt"); _t.save(out["best_state"], p[:-5] + "_best.pt")
                rec = {"variant": vname, "fold": k, "seed": ts, "dataset": "zenodo_bio" + ("" if a.base else "_more"), "win": 10.0,
                       "stride": 2.0, "minutes": (time.time() - t0)/60, **extra,
                       "y_val": y[va].tolist(), "vid_val": V[va].tolist(), "y_te": y[te].tolist(), "vid_te": V[te].tolist(),
                       "n_train": int(len(tr)), "n_train_extra": int((~E[tr]).sum())}
                for nm, (Pv, Pt) in ent.items():
                    rec[nm] = {"p_val": np.round(Pv, 6).tolist(), "p_te": np.round(Pt, 6).tolist(),
                               "metrics": [B.bio_metrics(Pv, y[va], V[va], Pt, y[te], V[te], n) for n in (1, 3, 5)]}
                json.dump(rec, open(p + ".tmp", "w")); os.replace(p + ".tmp", p)
                m = rec["last"]["metrics"][2]
                print(f"{vname:22s} f{k} ts{ts} | last 5-win EER {100*m['eer']:.2f}% top1 {m['top1']:.3f} | "
                      f"1-win EER {100*rec['last']['metrics'][0]['eer']:.2f}% | {rec['minutes']:.1f} min", flush=True)
    print("STEP11_DONE")

if __name__ == "__main__": main()
