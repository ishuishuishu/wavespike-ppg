# D7 CinC 2015 sanity task (optional)
CINC_BASE = "https://physionet.org/files/challenge-2015/1.0.0/"
CINC_TYPES = {"tachycardia": "Tachycardia", "ventricular_tachycardia": "Ventricular_Tachycardia"}

def _norm(s): return re.sub(r"[^a-z]+", "_", str(s).lower()).strip("_")

def load_cinc2015(cfg):
    import wfdb
    raw = f"{DATA_ROOT}/raw/cinc2015"; cache = f"{DATA_ROOT}/processed/cinc2015_tachy_vt_pleth_{cfg.fs}hz_30s.npz"
    if os.path.exists(cache):
        d = np.load(cache); print("loaded cache", cache); return d["X"], d["y"], d["rec"].astype(str), d["atype"].astype(str)
    if fetch_physionet(["training/RECORDS"], CINC_BASE, raw, None): raise RuntimeError("training/RECORDS not found")
    recs = [l.strip() for l in open(f"{raw}/training/RECORDS") if l.strip()]
    fetch_physionet([f"training/{r}.hea" for r in recs], CINC_BASE, raw, None, desc="CinC headers")
    sel, unparsed = [], []
    for r in recs:
        if not _nonempty(f"{raw}/training/{r}.hea"): continue
        hd = wfdb.rdheader(f"{raw}/training/{r}"); com = [_norm(c) for c in (hd.comments or [])]
        atype = next((CINC_TYPES[c] for c in com if c in CINC_TYPES), None)
        known_type = any(c in {"asystole", "bradycardia", "ventricular_flutter_fib"} | set(CINC_TYPES) for c in com)
        truth = 1 if "true_alarm" in com else 0 if "false_alarm" in com else None
        if truth is None or not known_type: unparsed.append((r, hd.comments)); continue
        if atype and any("pleth" in s.lower() for s in hd.sig_name): sel.append((r, atype, truth, hd))
    print(f"D7 headers: {len(recs)} records, {len(unparsed)} unparsed, {len(sel)} Tachycardia/VT with PLETH. "
          f"Example comments: {wfdb.rdheader(f'{raw}/training/{recs[0]}').comments}")
    if len(unparsed) == len(recs):
        raise StopForUser(f"No alarm type / true-false label found in the header comments (e.g. {unparsed[0]}). Adapt load_cinc2015.")
    files = sorted({f"training/{f}" for _, _, _, hd in sel for f in set(hd.file_name)})
    fetch_physionet(files, CINC_BASE, raw, None, desc="CinC signals")
    Xc, yc, rc, ac = [], [], [], []
    for r, atype, truth, hd in sel:
        rec = wfdb.rdrecord(f"{raw}/training/{r}"); ch = next(i for i, s in enumerate(rec.sig_name) if "pleth" in s.lower())
        fs = int(round(rec.fs)); x = rec.p_signal[int(270*fs):int(300*fs), ch]
        if len(x) < 30*fs: continue
        x, why = clean_segment(x, fs)
        if x is None: continue
        g = math.gcd(fs, cfg.fs); Xc.append(resample_poly(x, cfg.fs//g, fs//g).astype(np.float32)); yc.append(truth); rc.append(r); ac.append(atype)
    Xc, yc, rc, ac = np.stack(Xc), np.array(yc), np.array(rc), np.array(ac)
    np.savez_compressed(cache, X=Xc, y=yc, rec=rc, atype=ac); return Xc, yc, rc, ac

if RUN_CINC2015:
    Xc, yc, rc, ac = load_cinc2015(CFG); Xc = normalize_windows(Xc)
    print(f"D7: {len(yc)} windows | true alarms {int(yc.sum())} / false {int((1-yc).sum())} | by type {Counter(ac)}")
    cfg_c = variant_config("full", 0); cfg_c.n_classes = 2
    folds_c = make_folds(yc, rc, 5, seed=0); Fc = feature_matrix(Xc, CFG.fs)
    P_snn, P_lr, idx_all = [], [], []
    for k, (tr_c, va_c, te_c) in enumerate(folds_c):
        o = train(cfg_c, tr_c, va_c, verbose=False, Xa=Xc, ya=yc)
        P_snn.append(predict(o["best"], te_c, Xa=Xc))
        trva = np.concatenate([tr_c, va_c]); P_lr.append(IrregLR(2).fit(Fc[trva], yc[trva]).predict_proba(Fc[te_c])); idx_all.append(te_c)
    idx_all = np.concatenate(idx_all)
    D7_RESULT = {name: {"oof_auroc": _auc(yc[idx_all], np.concatenate(P)[:, 1]),
                        "oof_f1": metrics(yc[idx_all], np.concatenate(P))["macro_f1"]} for name, P in (("full", P_snn), ("irreg_lr", P_lr))}
    os.makedirs(f"{DATA_ROOT}/results/cinc2015_v3", exist_ok=True)
    json.dump(D7_RESULT, open(f"{DATA_ROOT}/results/cinc2015_v3/oof.json", "w"))
    print("D7 CinC 2015 Tachy/VT true-vs-false alarm (NOT AF detection), pooled OOF, 1 fold seed x 1 training seed:", D7_RESULT)
else:
    print("D7 skipped (RUN_CINC2015 = False)")
