#!/usr/bin/env python3
"""Step 11 aggregation (STEP11_PREDECLARED_PROTOCOL.md). Recomputes every endpoint from the saved probabilities with
bio_run.bio_metrics, checks it against the metrics stored in each run's JSON, then applies the predeclared verdicts.

Usage: python -I step11_aggregate.py <teammate_reference_metrics.json> <out summary json>
Reads $WAVESPIKE_DATA_ROOT/results/bio_zenodo_bio_w10 (Part A) and .../step11_more_w10 (Part B).
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.environ.get("WAVESPIKE_REPO", os.path.join(os.path.expanduser("~"), "capstone_metrics_run", "wavespike-ppg")), "biometrics"))
import bio_run as B
from scipy import stats

KEYS = ("eer", "eer_val", "top1", "macro_f1", "far", "frr", "auc", "tau", "n_decisions")
MAXDEV = {}                 # per metric: largest |recomputed - stored| over all runs, checkpoints and n
FLIPS = {"far": 0, "frr": 0, "top1": 0}   # number of (run, checkpoint, n) cases where far/frr/top1 differ at all
NS = (1, 3, 5)

def load_runs(d, v):
    out = {}
    for p in sorted(glob.glob(f"{d}/{v}_f*_ts*.json")):
        r = json.load(open(p)); k, s = r["fold"], r["seed"]
        y_va, v_va, y_te, v_te = map(np.asarray, (r["y_val"], r["vid_val"], r["y_te"], r["vid_te"]))
        rec = {"minutes": r["minutes"], "params": r.get("params"), "n_train": r.get("n_train"), "n_train_extra": r.get("n_train_extra"),
               "epochs_run": r.get("epochs_run"), "hid_rate_last": r.get("hid_rate_last")}
        for ck in ("last", "best"):
            Pv, Pt = np.asarray(r[ck]["p_val"]), np.asarray(r[ck]["p_te"])
            mets = [B.bio_metrics(Pv, y_va, v_va, Pt, y_te, v_te, n) for n in NS]
            # endpoint cross-check vs the run's own JSON. Saved probabilities are rounded to 6 decimals; with n > 1 the
            # decision score is a mean of log-probabilities, so rounding small p moves scores and tau by up to ~1e-3 and
            # can flip a trial. Tolerance (fixed 2026-11-23, note N11): the first cap of 1e-3 on every key is kept, except
            # for frr and top1, whose denominators are so small (240 decisions at n=5) that ONE decision is 4.2e-3; those
            # two may differ by at most one decision. Deviations and one-trial flip counts are recorded in the summary.
            K = Pt.shape[1]
            for m_new, m_old in zip(mets, r[ck]["metrics"]):
                nd = m_new["n_decisions"]; assert nd == m_old["n_decisions"], (p, ck, "n_decisions")
                tol = {"far": 1e-3, "frr": max(1e-3, 1/nd), "top1": max(1e-3, 1/nd), "eer": 1e-3, "eer_val": 1e-3, "macro_f1": 1e-3, "tau": 1e-3}
                for kk, t in tol.items():
                    dev = abs(m_new[kk] - m_old[kk]); MAXDEV[kk] = max(MAXDEV.get(kk, 0.0), dev)
                    assert dev <= t + 1e-12, (p, ck, kk, m_new[kk], m_old[kk])
                    if kk in FLIPS and dev > 1e-12: FLIPS[kk] += 1
            rec[ck] = [{kk: m[kk] for kk in KEYS} for m in mets]
        out[(k, s)] = rec
    return out

def summ(runs, ck="last"):
    s = {}
    for j, n in enumerate(NS):
        s[f"n{n}"] = {kk: {"mean": float(np.mean([r[ck][j][kk] for r in runs.values()])),
                           "sd": float(np.std([r[ck][j][kk] for r in runs.values()], ddof=1)) if len(runs) > 1 else 0.0,
                           "min": float(np.min([r[ck][j][kk] for r in runs.values()])),
                           "max": float(np.max([r[ck][j][kk] for r in runs.values()]))} for kk in KEYS}
    s["runs"] = len(runs); s["per_run_eer5"] = {f"f{k}_ts{t}": runs[(k, t)][ck][2]["eer"] for (k, t) in sorted(runs)}
    s["per_run_top1_5"] = {f"f{k}_ts{t}": runs[(k, t)][ck][2]["top1"] for (k, t) in sorted(runs)}
    s["minutes_mean"] = float(np.mean([r["minutes"] for r in runs.values()]))
    s["params"] = next(iter(runs.values()))["params"]
    s["n_train_mean"] = float(np.mean([r["n_train"] for r in runs.values()])) if next(iter(runs.values()))["n_train"] else None
    return s

def paired(a, b, j=2, kk="eer"):
    keys = sorted(set(a) & set(b)); d = np.array([a[x]["last"][j][kk] - b[x]["last"][j][kk] for x in keys])
    if len(d) > 1:
        h = stats.t.ppf(0.975, len(d) - 1)*d.std(ddof=1)/np.sqrt(len(d)); ci = [float(d.mean() - h), float(d.mean() + h)]
    else: ci = [float("nan")]*2
    return {"n_pairs": len(d), "mean_diff": float(d.mean()), "ci95": ci, "wins_first_lower": int((d < 0).sum()),
            "per_pair": {f"f{k}_ts{t}": float(x) for (k, t), x in zip(keys, d)}}

def main():
    ref = json.load(open(sys.argv[1]))["bio_zenodo_bio_w10"]; root = os.environ["WAVESPIKE_DATA_ROOT"] + "/results"
    A = {v: load_runs(f"{root}/bio_zenodo_bio_w10", v) for v in ("x_tau_amp", "cwt_cnn_amp", "beat_svm_amp", "dwt_svm_amp")}
    Bv = {v: load_runs(f"{root}/step11_more_w10", v) for v in ("x_tau_amp_more", "cwt_cnn_amp_more", "x_h128_tau_amp_more")}
    out = {"partA": {}, "partB": {}, "checks": "every endpoint recomputed from the saved p_val/p_te (rounded to 6 decimals) and matched to each run's JSON: every metric within 1e-3 except frr and top1, which may differ by one decision (1/240 at n=5); per-metric max deviation in checks_max_endpoint_deviation and the number of (run, checkpoint, n) cases with any far/frr/top1 difference in checks_one_trial_flips (note N11)"}
    for v, runs in A.items():
        if not runs: out["partA"][v] = {"runs": 0}; continue
        s = summ(runs); r = ref[v]["last"]
        delta = {f"n{n}": {kk: s[f"n{n}"][kk]["mean"] - r[j][kk]["mean"] for kk in ("eer", "top1", "macro_f1", "far", "frr")} for j, n in enumerate(NS)}
        d5 = abs(delta["n5"]["eer"])*100
        if v in ("beat_svm_amp", "dwt_svm_amp"): verdict = "exact (|d| <= 0.3 pp)" if d5 <= 0.3 else "not exact"
        else: verdict = "corroborated" if d5 <= 1.5 else ("partly corroborated" if d5 <= 3 else "not corroborated")
        out["partA"][v] = {**s, "reference_last": r, "reference_runs": ref[v]["runs"], "delta_vs_reference": delta,
                           "verdict_5win_eer": verdict, "complete": len(runs) == ref[v]["runs"]}
    for v, runs in Bv.items(): out["partB"][v] = summ(runs) if runs else {"runs": 0}
    # selection among the SNN candidates by mean validation 5-window EER (last checkpoint)
    # erratum E2 (2026-11-23): B3 (x_h128_tau_amp_more) dropped on compute grounds; candidates are x_tau_amp and B1
    cand = {"x_tau_amp": A["x_tau_amp"], "x_tau_amp_more": Bv["x_tau_amp_more"]}
    val = {v: float(np.mean([r["last"][2]["eer_val"] for r in runs.values()])) for v, runs in cand.items() if len(runs) == 6}
    out["selection"] = {"val_eer5_mean": val, "complete": len(val) == len(cand), "selected": min(val, key=val.get) if len(val) == len(cand) else None}
    sel = out["selection"]["selected"]
    if sel:
        S = cand[sel]; bench = A["cwt_cnn_amp"]
        pb = paired(S, bench) if bench else None
        if pb and pb["n_pairs"] == 6:
            vd = "beats" if pb["wins_first_lower"] == 6 else ("matches" if abs(pb["mean_diff"])*100 <= 1.0 else "does not match")
        else: vd = "incomplete"
        out["benchmark"] = {"selected": sel, "vs_our_cwt_cnn_amp": pb, "verdict": vd,
                            "vs_reported_13.98": float(np.mean([r["last"][2]["eer"] for r in S.values()]) - ref["cwt_cnn_amp"]["last"][2]["eer"]["mean"]),
                            "vs_cwt_cnn_amp_more": paired(S, Bv["cwt_cnn_amp_more"]) if Bv["cwt_cnn_amp_more"] else None,
                            "vs_x_tau_amp_partA": paired(S, A["x_tau_amp"]) if sel != "x_tau_amp" else None,
                            "top1_vs_our_cwt_cnn_amp": paired(S, bench, kk="top1") if bench else None}
        # also: does more data help the benchmark itself?
        if Bv["cwt_cnn_amp_more"] and bench: out["benchmark"]["cwt_more_vs_cwt"] = paired(Bv["cwt_cnn_amp_more"], bench)
        # project goal at the teammate's threshold (2 % FAR on validation): FAR < 3 % and FRR < 5 %
        out["project_goal"] = {f"f{k}_ts{t}": {"far": r["last"][2]["far"], "frr": r["last"][2]["frr"],
                                               "met": r["last"][2]["far"] < 0.03 and r["last"][2]["frr"] < 0.05} for (k, t), r in sorted(S.items())}
    out["checks_max_endpoint_deviation"] = MAXDEV; out["checks_one_trial_flips"] = FLIPS
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("selection",)}, indent=1))
    for part in ("partA", "partB"):
        for v, s in out[part].items():
            if s.get("runs"):
                m = s["n5"]; print(f"{part} {v:22s} runs {s['runs']} | EER5 {100*m['eer']['mean']:.2f}±{100*m['eer']['sd']:.2f} | top1-5 {100*m['top1']['mean']:.1f} | "
                                   f"EER1 {100*s['n1']['eer']['mean']:.2f} | valEER5 {100*m['eer_val']['mean']:.2f}" + (f" | {s['verdict_5win_eer']}" if "verdict_5win_eer" in s else ""))
    if "benchmark" in out: print("benchmark:", out["benchmark"]["selected"], out["benchmark"]["verdict"], out["benchmark"]["vs_our_cwt_cnn_amp"])

if __name__ == "__main__": main()
