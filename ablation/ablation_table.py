#!/usr/bin/env python3
"""Builds the wavelet-ablation table from our earlier runs plus any new runs on this machine.

Usage: python -I ablation_table.py [--n 5] [--checkpoint last]

Sources, merged per run (dataset, variant, fold, seed):
  1. ablation_runs_so_far.csv  - metrics of the 75 runs already done on the original machine (shipped with this folder)
  2. $WAVESPIKE_DATA_ROOT/results/<folder>/<variant>_f<fold>_ts<seed>.json - runs done here (they win if a run is in both)

Prints, for EER and top-1: mean over runs, number of runs (out of 6) and the range; then paired comparisons
(same fold and seed) of each front end against the fully learnable wavelet, counting wins per repeat.
"""
import argparse, csv, glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
FOLDERS = {"zenodo": "bio_zenodo_bio_w10", "seeing_red": "bio_seeing_red_luma_w10_trs0.5"}
ROWS = [("No wavelet", "x_raw_tau_amp", "x_aug_raw_tau_amp"),
        ("Fixed db4", "x_fixed_tau_amp", "x_aug_fixed_tau_amp"),
        ("Partly learnable (one shared filter)", "x_shared_tau_amp", "x_aug_shared_tau_amp"),
        ("Fully learnable (filter per level)", "x_tau_amp", "x_aug_tau_amp")]
FULL = {False: "x_tau_amp", True: "x_aug_tau_amp"}


def load(n, ck):
    runs = {}                                               # (ds, variant, fold, seed) -> {"eer", "top1", "src"}
    with open(os.path.join(HERE, "ablation_runs_so_far.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if int(r["n_windows"]) == n and r["checkpoint"] == ck:
                runs[(r["dataset"], r["variant"], int(r["fold"]), int(r["seed"]))] = {
                    "eer": float(r["eer"]), "top1": float(r["top1"]), "src": "ours"}
    root = os.environ.get("WAVESPIKE_DATA_ROOT")
    new = 0
    if root:
        for ds, folder in FOLDERS.items():
            for p in glob.glob(f"{root}/results/{folder}/x_*_f*_ts*.json"):
                r = json.load(open(p)); m = r[ck]["metrics"][(1, 3, 5).index(n)]
                runs[(ds, r["variant"], int(r["fold"]), int(r["seed"]))] = {"eer": m["eer"], "top1": m["top1"], "src": "new"}
                new += 1
    return runs, new


def cell(runs, ds, v, key):
    xs = [x[key] for (d, vv, _, _), x in runs.items() if d == ds and vv == v]
    if not xs: return "missing (0/6)"
    return f"{100*sum(xs)/len(xs):5.1f}  ({len(xs)}/6, {100*min(xs):.1f}-{100*max(xs):.1f})"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=5, choices=(1, 3, 5))
    ap.add_argument("--checkpoint", default="last", choices=("last", "best")); a = ap.parse_args()
    runs, new = load(a.n, a.checkpoint)
    print(f"{len(runs)} runs ({new} result files found on this machine). Decisions on {a.n} windows, {a.checkpoint} epoch.\n")
    for key, title in (("eer", "EER % (lower is better)"), ("top1", "Top-1 accuracy % (higher is better)")):
        print(f"=== {title}: mean (runs, min-max) ===")
        for ds in FOLDERS:
            print(f"\n{ds}")
            print(f"  {'Front end':38s} | {'no augmentation':26s} | {'with augmentation':26s}")
            for name, v0, v1 in ROWS:
                print(f"  {name:38s} | {cell(runs, ds, v0, key):26s} | {cell(runs, ds, v1, key):26s}")
        print()
    print("=== Paired EER vs the fully learnable wavelet, same augmentation, same fold and seed ===")
    print("    'learnable better in k of m' = repeats where the fully learnable wavelet had the lower EER.")
    print("    Project rule: a difference is only claimed if it holds in every repeat (m = 6).")
    for ds in FOLDERS:
        print(f"\n{ds}")
        for aug in (False, True):
            for name, v0, v1 in ROWS[:-1]:
                v = v1 if aug else v0; f = FULL[aug]
                pairs = [(runs[(ds, v, k, s)]["eer"], runs[(ds, f, k, s)]["eer"]) for k in range(3) for s in range(2)
                         if (ds, v, k, s) in runs and (ds, f, k, s) in runs]
                if not pairs: print(f"  {'aug' if aug else 'no aug':6s} {name:38s}: no paired repeats yet"); continue
                wins = sum(fl < o for o, fl in pairs); d = 100*sum(fl - o for o, fl in pairs)/len(pairs)
                verdict = ("learnable better in every repeat" if wins == len(pairs) == 6 else
                           "learnable worse in every repeat" if wins == 0 and len(pairs) == 6 and all(fl > o for o, fl in pairs) else
                           "no consistent difference" if len(pairs) == 6 else "too few repeats")
                print(f"  {'aug' if aug else 'no aug':6s} {name:38s}: learnable better in {wins} of {len(pairs)}, "
                      f"mean EER difference {d:+.1f} points -> {verdict}")


if __name__ == "__main__":
    main()
