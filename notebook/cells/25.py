# §9c Results: pooled out-of-fold metrics per repeat -> mean ± SD over repeats, paired differences vs "full"
def load_runs(res_dir=None):
    R = []
    for p in glob.glob(f"{res_dir or RES_DIR}/*_cv*_ts*_fold*.json"):
        r = json.load(open(p))
        if r.get("data_fp") == DATA_FP: R.append(r)
    return R

def pooled_table(R, ckpt):
    by = {}
    for r in R: by.setdefault((r["variant"], r["cv_seed"], r["train_seed"]), []).append(r)
    rows = []
    for (v, cs, ts), rr in sorted(by.items()):
        if len({r["fold"] for r in rr}) < CFG.n_folds: continue           # incomplete repeat
        rr = sorted(rr, key=lambda r: r["fold"])
        idx = np.concatenate([r["te_idx"] for r in rr]); P = np.concatenate([np.array(r[ckpt]["prob"]) for r in rr]); P = P/P.sum(1, keepdims=True)   # undo rounding drift
        assert np.array_equal(np.sort(idx), np.arange(len(y))), "pooled OOF predictions must cover every window exactly once"
        mm = metrics(y[idx], P)
        row = {"variant": v, "cv_seed": cs, "train_seed": ts, "oof_auroc": mm["auroc"], "oof_f1": mm["macro_f1"],
               "fold_auroc_mean": float(np.nanmean([r[ckpt]["auroc"] for r in rr])),
               "fold_auroc_sd": float(np.nanstd([r[ckpt]["auroc"] for r in rr], ddof=1)),
               "epochs": float(np.mean([r["epochs_run"] for r in rr]))}
        if AF_IDX is not None:
            row["oof_subj_auroc"] = subject_level(P[:, AF_IDX], (y[idx] == AF_IDX).astype(int), groups[idx])[0]
        if all("ext" in r[ckpt] for r in rr):
            for k_ in ("auroc", "subj_auroc", "macro_f1"):
                row["ext_" + k_] = float(np.nanmean([r[ckpt]["ext"][k_] for r in rr]))
        rows.append(row)
    return pd.DataFrame(rows)

def _ms(s):
    s = s.dropna()
    return "—" if len(s) == 0 else (f"{s.mean():.3f} ± {s.std(ddof=1):.3f}" if len(s) > 1 else f"{s.mean():.3f}")

def summarize(df, metric_cols, delta_cols=("oof_auroc", "oof_subj_auroc"), ref="full"):
    out = []
    for v in [v for v in VARIANTS if v in set(df.variant)]:
        d = df[df.variant == v]; row = {"variant": v, "repeats": len(d)}
        for c in metric_cols: row[c] = _ms(d[c]) if c in d else "—"
        if v != ref and ref in set(df.variant):
            mg = d.merge(df[df.variant == ref], on=["cv_seed", "train_seed"], suffixes=("", "_ref"))
            for c in delta_cols:
                if c not in mg: continue
                diff = (mg[c] - mg[c + "_ref"]).dropna().values
                row[f"Δ{c} vs {ref}"] = _ms(pd.Series(diff))
                if c == delta_cols[0]:
                    n, pos, neg = len(diff), int((diff > 0).sum()), int((diff < 0).sum())
                    p = wilcoxon(diff).pvalue if n >= 6 and np.any(diff != 0) else float("nan")
                    row[f"better/worse ({c})"] = f"{pos}/{neg} of {n}"
                    row["wilcoxon p"] = f"{p:.3f}" if p == p else "—"
                    row["verdict"] = ("too few repeats (<6)" if n < 6 else "BETTER in every repeat" if pos == n else
                                      "WORSE in every repeat" if neg == n else "no consistent difference")
        out.append(row)
    return pd.DataFrame(out).fillna("—")

R_ALL = load_runs()
int_cols = ["oof_auroc", "oof_f1", "oof_subj_auroc", "fold_auroc_mean", "epochs"]
ext_cols = ["ext_auroc", "ext_subj_auroc", "ext_macro_f1"]
EXT_DELTAS = ("ext_auroc", "ext_subj_auroc")
TABLES = {}
for ckpt in ("best", "last"):
    df = pooled_table(R_ALL, ckpt)
    if df.empty: print(f"[{ckpt}] no complete repeat yet"); continue
    TABLES[ckpt] = summarize(df, [c for c in int_cols if c in df.columns])
    df.to_csv(f"{RES_DIR}/per_repeat_{ckpt}.csv", index=False)
    print(f"\n=== {DATASET} | checkpoint: {ckpt} | internal {CFG.n_folds}-fold out-of-fold | mean ± SD over repeats ===")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(TABLES[ckpt].to_string(index=False))
    if XE is not None and "ext_auroc" in df:
        T_ext = summarize(df, ext_cols, EXT_DELTAS)
        if EXTERNAL_OVERLAP_STATUS == "verified":
            TABLES[ckpt + " | external MIMIC PERform AF (fold-model average)"] = T_ext
            print(f"\n=== external MIMIC PERform AF (overlap check verified) | checkpoint: {ckpt} ===")
        else:
            print(f"\n⚠ external MIMIC PERform AF [{EXTERNAL_OVERLAP_STATUS}] — NOT part of the results table:")
        with pd.option_context("display.width", 250, "display.max_columns", None):
            print(T_ext.to_string(index=False))

if TABLES:
    with open(f"{RES_DIR}/results_table.md", "w") as f:
        f.write(f"# WaveSpike-PPG v3 results: {DATASET}\n\nData fingerprint `{DATA_FP[:8]}`, {len(np.unique(groups))} subjects, "
                f"{len(y)} windows. Repeats = fold seeds {CV_SEEDS} × training seeds {TRAIN_SEEDS}. External overlap check: "
                f"{EXTERNAL_OVERLAP_STATUS}.\nImprovement is claimed only if the paired Δ has the same sign in every repeat (≥ 6).\n\n")
        for ckpt, T_ in TABLES.items():
            try: body = T_.to_markdown(index=False)
            except ImportError: body = "```\n" + T_.to_string(index=False) + "\n```"
            f.write(f"## Checkpoint: {ckpt}\n\n" + body + "\n\n")
    for ckpt, T_ in TABLES.items(): T_.to_csv(f"{RES_DIR}/results_table_{re.sub(r'[^A-Za-z0-9]+', '_', ckpt).strip('_')}.csv", index=False)
    print("\nwritten:", f"{RES_DIR}/results_table.md")
