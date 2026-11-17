# Load (Drive cache when available) -> D6 overlap check -> normalise -> D5 subject-wise folds
def load_cached(name, loader, cfg, **kw):
    cache = f"{DATA_ROOT}/processed/{name}_{cfg.fs}hz_{cfg.win_sec}s.npz"
    if os.path.exists(cache) and name != "synthetic":
        d = np.load(cache, allow_pickle=True); print("loaded cache", cache)
        return d["X"], d["y"], d["subject"], [str(c) for c in d["class_names"]]
    X_, y_, g_, names = loader(cfg, **kw)
    if name != "synthetic":
        np.savez_compressed(cache, X=X_, y=y_, subject=g_, class_names=np.array(names)); print("cached to", cache)
    return X_, y_, g_, names

def normalize_windows(X):
    mu = X.mean(1, keepdims=True); sd = X.std(1, keepdims=True) + 1e-6
    return ((X - mu)/sd).astype(np.float32)

# ---- D6 part 1: MIMIC-III subject IDs of the external set (must run before Ext-PPG subjects are picked) ----
EXCLUDE_IDS, EXTERNAL_OVERLAP_STATUS = None, "n/a (no external set)"
XE = yE = gE = None
if EXTERNAL_DATASET == "mimic_perform_af":
    _ids = mimic_perform_subject_ids()
    _with = {k: v["ids"] for k, v in _ids.items() if v["ids"]}
    _first = next(iter(_ids.values()))["header"]
    print("D6 first MIMIC PERform AF header (searched for an ID):\n   " + "\n   ".join(_first.splitlines()[:12]))
    print(f"D6 MIMIC-III subject ID found in {len(_with)}/{len(_ids)} MIMIC PERform AF records, e.g. "
          f"{dict(list(_with.items())[:5])}")
    EXCLUDE_IDS = sorted({i for v in _with.values() for i in v})
    if len(_with) == len(_ids):
        EXTERNAL_OVERLAP_STATUS = "verified"
    else:
        EXTERNAL_OVERLAP_STATUS = (f"UNVERIFIED: no MIMIC-III subject ID in the name or header of {len(_ids)-len(_with)}"
                                   f"/{len(_ids)} MIMIC PERform AF records")
        print("⚠" * 3, "D6:", EXTERNAL_OVERLAP_STATUS, "-> overlap with Ext-PPG can't be fully excluded. External "
              "results are labelled UNVERIFIED and kept out of the main table.")
    XE, yE, gE, CLASS_NAMES_E = load_cached("mimic_perform_af", load_mimic_perform_af, CFG)
    XE = normalize_windows(XE)
    print(f"external MIMIC PERform AF: {XE.shape[0]} windows, {len(np.unique(gE))} subjects, classes "
          f"{dict(zip(CLASS_NAMES_E, np.bincount(yE).tolist()))}")

# ---- training data ----
loader, CFG.win_sec = LOADERS[DATASET]
if DATASET == "mimic3_ext_ppg":
    X, y, groups, RECORD_IDS, CLASS_NAMES = load_mimic3_ext_ppg(CFG, EXCLUDE_IDS)
else:
    X, y, groups, CLASS_NAMES = load_cached(DATASET, loader, CFG, **(SYNTH_KW if DATASET == "synthetic" else {}))
    RECORD_IDS = np.array([f"{g}#{i}" for i, g in enumerate(groups)])
groups = np.asarray(groups).astype(str)

# ---- D6 part 2: assert there is no overlap left in the training data ----
if EXCLUDE_IDS is not None:
    _num = np.array([subject_num(g) for g in groups], dtype=object)
    _ov = np.array([n is not None and n in set(EXCLUDE_IDS) for n in _num])
    print(f"D6 Ext-PPG subjects overlapping MIMIC PERform AF in the training data: {len(set(groups[_ov]))} -> dropped")
    if _ov.any(): X, y, groups, RECORD_IDS = X[~_ov], y[~_ov], groups[~_ov], RECORD_IDS[~_ov]
    if any(n is None for n in _num):
        EXTERNAL_OVERLAP_STATUS = "UNVERIFIED: some Ext-PPG subject IDs have no single number"
        print("⚠ D6:", EXTERNAL_OVERLAP_STATUS)

X = normalize_windows(X); CFG.n_classes = len(CLASS_NAMES); T_WIN = X.shape[1]; CFG.win_sec = T_WIN // CFG.fs
AF_IDX = CLASS_NAMES.index("AF") if "AF" in CLASS_NAMES else None
print(f"{DATASET}: X {X.shape} | subjects {len(np.unique(groups))} | segments per class "
      f"{dict(zip(CLASS_NAMES, np.bincount(y, minlength=CFG.n_classes).tolist()))}")

def subject_strat_labels(y, groups, af_idx):
    # D5: stratify on the SUBJECT's AF status (any AF segment). Without an AF class: the subject's majority label.
    u, inv = np.unique(groups, return_inverse=True)
    if af_idx is not None:
        s = np.zeros(len(u), int); np.maximum.at(s, inv, (y == af_idx).astype(int)); return s[inv]
    cnt = np.zeros((len(u), y.max()+1), int); np.add.at(cnt, (inv, y), 1); return cnt.argmax(1)[inv]

STRAT = subject_strat_labels(y, groups, AF_IDX)
if AF_IDX is not None:
    _sa = pd.Series(STRAT).groupby(groups).first()
    print(f"subjects with any AF segment: {int(_sa.sum())} / {len(_sa)}")
DATA_FP = hashlib.sha1((DATASET + "|" + "|".join(RECORD_IDS.astype(str)) + "|" + y.tobytes().hex()).encode()).hexdigest()

def make_folds(y, groups, n_folds, seed=0):
    # Stratified subject-wise outer folds; inside each training part, one stratified group fold becomes validation.
    # `y` here is the stratification label (the subject's AF status), not the segment label.
    outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for trva, te in outer.split(np.zeros(len(y)), y, groups):
        inner = StratifiedGroupKFold(n_splits=max(3, n_folds-1), shuffle=True, random_state=seed+1)
        tr_i, va_i = next(inner.split(np.zeros(len(trva)), y[trva], groups[trva]))
        tr, va = trva[tr_i], trva[va_i]
        assert not (set(groups[tr]) & set(groups[te])) and not (set(groups[va]) & set(groups[te])) and not (set(groups[tr]) & set(groups[va]))
        folds.append((tr, va, te))
    assert np.array_equal(np.sort(np.concatenate([f[2] for f in folds])), np.arange(len(y))), "test folds must partition the data"
    return folds

FOLDS_BY_CV = {cs: make_folds(STRAT, groups, CFG.n_folds, seed=cs) for cs in CV_SEEDS}
FOLDS = FOLDS_BY_CV[CV_SEEDS[0]]
for k, (tr, va, te) in enumerate(FOLDS):
    print(f"fold {k}: train {len(tr)} ({len(set(groups[tr]))} subj) | val {len(va)} ({len(set(groups[va]))}) | test {len(te)} "
          f"({len(set(groups[te]))}) | test classes {np.bincount(y[te], minlength=CFG.n_classes)}")

fig, ax = plt.subplots(CFG.n_classes, 1, figsize=(12, 1.8*CFG.n_classes), sharex=True)
for c in range(CFG.n_classes):
    i = np.where(y == c)[0][0]; ax[c].plot(np.arange(T_WIN)/CFG.fs, X[i], lw=.8); ax[c].set_ylabel(CLASS_NAMES[c])
ax[-1].set_xlabel("s"); plt.tight_layout(); plt.show()
