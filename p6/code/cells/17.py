# Experiment definitions shared by §6 and §9: variants, one run = (variant, fold seed, training seed, fold)
VARIANTS = {
    "full":          {},                                            # v3 defaults (ht_nonneg=True)
    "fixed_wavelet": {"learn_wavelet": False},
    "shared_filter": {"per_level": False},
    "no_ht":         {"use_ht": False},
    # P0.4 baselines
    "irreg_lr":      {"_kind": "lr"},                               # (a) interval irregularity + logistic regression
    "gru_twin":      {"arch": "gru"},                               # (b) same front end + GRU of matched size
    "snn_raw":       {"use_wavelet": False, "use_ht": False},       # (c) SNN without wavelet
    # P1 fixes, one flag each
    "ht_despawn":    {"ht_nonneg": False},                          # P1.6 the unconstrained v2 HT
    "lean":          {"drop_bands": [0, 1], "snn_decim": 2},        # P1.7
    "aug":           {"augment": True},                             # P1.8
    "spkdrop":       {"spike_dropout": 0.15},                       # P1.8
    "h32":           {"hidden": 32},                                # P1.8
    "reg_all":       {"augment": True, "spike_dropout": 0.15, "hidden": 32},
    "full_clean":    {"_clean": True},                              # P1.9 train/val cleaned, test untouched
    # v3.1 accuracy candidates (all new, defaults unchanged). Screened on fold seed 0 / train seed 0 only.
    "tau_het":       {"tau_het": True},                             # hidden leaks spread over 0.2-4 s (can remember ~1-2 beats)
    "augplus":       {"augment": True, "augment_plus": True},       # time-warp + amplitude modulation on top of aug
    "slowtrain":     {"epochs": 40, "patience": 40, "lr": 1e-3},    # no early stop; cosine to zero over 40 epochs
    "wavlr":         {"lr_wavelet": 5e-3, "lambda_wav": 0.1},       # let the wavelet filters move 10x more
    "raw_tau":       {"use_wavelet": False, "use_ht": False, "tau_het": True},
    "tau_aug":       {"tau_het": True, "augment": True, "augment_plus": True},
    "tau_aug_slow":  {"tau_het": True, "augment": True, "augment_plus": True, "epochs": 40, "patience": 40, "lr": 1e-3},
    "lean_tau_aug":  {"drop_bands": [0, 1], "snn_decim": 2, "tau_het": True, "augment": True, "augment_plus": True},
    # stacked on the best dev-screen variant (tau_aug_slow)
    "raw_tau_aug_slow":  {"use_wavelet": False, "use_ht": False, "tau_het": True, "augment": True, "augment_plus": True, "epochs": 40, "patience": 40, "lr": 1e-3},
    "tas_spkdrop":   {"tau_het": True, "augment": True, "augment_plus": True, "epochs": 40, "patience": 40, "lr": 1e-3, "spike_dropout": 0.15},
    "tas_wavlr":     {"tau_het": True, "augment": True, "augment_plus": True, "epochs": 40, "patience": 40, "lr": 1e-3, "lr_wavelet": 5e-3, "lambda_wav": 0.1},
    "tas_60":        {"tau_het": True, "augment": True, "augment_plus": True, "epochs": 60, "patience": 60, "lr": 1e-3},
    "tas_h128":      {"tau_het": True, "augment": True, "augment_plus": True, "epochs": 40, "patience": 40, "lr": 1e-3, "hidden": 128},
    "lean_tas":      {"drop_bands": [0, 1], "snn_decim": 2, "tau_het": True, "augment": True, "augment_plus": True, "epochs": 40, "patience": 40, "lr": 1e-3},
}
EXPORTABLE = [v for v, s in VARIANTS.items() if s.get("arch", "snn") == "snn" and s.get("_kind") != "lr"]

RES_DIR = f"{DATA_ROOT}/results/{DATASET}_v3_{DATA_FP[:8]}"; CKPT_DIR = f"{RES_DIR}/ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)
print("results:", RES_DIR)

def variant_config(v, seed, base=None):
    c = copy.deepcopy(base or CFG); c.seed = seed
    for k, val in VARIANTS[v].items():
        if not k.startswith("_"): setattr(c, k, copy.deepcopy(val))
    return c

def run_path(v, cs, ts, k): return f"{RES_DIR}/{v}_cv{cs}_ts{ts}_fold{k}.json"

def _probs_entry(P, PE, te):
    r = metrics(y[te], P); r["prob"] = np.round(P, 5).tolist()
    if AF_IDX is not None: r["subj_auroc"] = subject_level(P[:, AF_IDX], (y[te] == AF_IDX).astype(int), groups[te])[0]
    if PE is not None and AF_IDX is not None:
        r["ext"] = binary_af_metrics(PE[:, AF_IDX], yE, gE); r["ext_p_af"] = np.round(PE[:, AF_IDX], 5).tolist()
    return r

def save_run(v, cs, ts, k, te, entries, extra):
    rec = {"variant": v, "cv_seed": cs, "train_seed": ts, "fold": k, "data_fp": DATA_FP, "te_idx": np.asarray(te).tolist(),
           "external_overlap": EXTERNAL_OVERLAP_STATUS, **entries, **extra}
    p = run_path(v, cs, ts, k); json.dump(rec, open(p + ".tmp", "w")); os.replace(p + ".tmp", p)   # atomic on Drive
    return rec

def run_one(v, cs, ts, k, verbose=False):
    tr, va, te = FOLDS_BY_CV[cs][k]; spec = VARIANTS[v]; t0 = time.time()
    if spec.get("_clean"):
        n0 = len(tr) + len(va); tr, va = tr[~label_disagrees(tr)], va[~label_disagrees(va)]
        if verbose: print(f"  cleaned train+val: {n0} -> {len(tr)+len(va)} windows (test untouched)")
    if spec.get("_kind") == "lr":
        trva = np.concatenate([tr, va]); lr_ = IrregLR(CFG.n_classes).fit(FEATS[trva], y[trva])
        P = lr_.predict_proba(FEATS[te]); PE = lr_.predict_proba(FEATS_E) if XE is not None else None
        e = _probs_entry(P, PE, te); entries = {"best": e, "last": e}; extra = {"epochs_run": 0, "best_epoch": -1, "hist": []}
    else:
        c = variant_config(v, ts); out = train(c, tr, va, verbose=verbose)
        tag = f"{CKPT_DIR}/{v}_cv{cs}_ts{ts}_fold{k}"
        torch.save(out["best_state"], tag + "_best.pt"); torch.save(out["last_state"], tag + "_last.pt")
        entries = {}
        for name in ("best", "last"):
            mdl = out[name]
            PE = predict(mdl, np.arange(len(XE)), Xa=XE) if XE is not None else None
            entries[name] = _probs_entry(predict(mdl, te), PE, te)
        extra = {"epochs_run": len(out["hist"]), "best_epoch": out["best_epoch"],
                 "hist": [{k_: (round(v_, 5) if isinstance(v_, float) else v_) for k_, v_ in h.items()} for h in out["hist"]]}
    extra["minutes"] = (time.time() - t0)/60
    return save_run(v, cs, ts, k, te, entries, extra)
