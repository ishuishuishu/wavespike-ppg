# MIMIC-III-Ext-PPG v1.1.0 (D1–D4). Moulaeifard, Charlton & Strodthoff, Sci Data 13, 668 (2026); doi:10.13026/r6k1-xt76
EXT_BASE = "https://physionet.org/files/mimic-iii-ext-ppg/1.1.0/"
EXT_RAW = f"{DATA_ROOT}/raw/mimic3_ext_ppg"
EXT_FS_NATIVE = 125
EXT_CLASSES = ["SR", "AF", "STACH", "SBRAD"] if EXT_MULTICLASS else ["SR", "AF"]     # AF is index 1 in both
PPG_NAMES = ("pleth", "ppg")

def subject_num(s):
    """MIMIC-III subject ID as an int ("p000123", "123" and 123 -> 123), or None if the value contains no digits."""
    m = re.findall(r"\d+", str(s))
    return int(m[0]) if len(m) == 1 else None          # several digit groups = ambiguous -> None (reported as unchecked)

def ext_d1_inspect(creds):
    os.makedirs(EXT_RAW, exist_ok=True)
    listing = physionet_listing(EXT_BASE, creds)
    print("D1 top-level entries of", EXT_BASE, ":", listing)
    readmes = [n for n in listing if n.lower().startswith("readme") and not n.endswith("/")]
    problems = ([] if "metadata.csv" in listing else ["metadata.csv is not in the top-level listing"]) + \
               ([] if readmes else ["there is no README* file in the top-level listing"])
    if problems:
        raise StopForUser("; ".join(problems) + ". Check the listing above and adapt ext_d1_inspect().")
    missing = fetch_physionet(readmes + ["metadata.csv"], EXT_BASE, EXT_RAW, creds, desc="README + metadata.csv")
    if missing: raise RuntimeError(f"download failed: {missing}")
    readme = "\n\n".join(f"===== {r} =====\n" + open(f"{EXT_RAW}/{r}", encoding="utf-8", errors="replace").read() for r in readmes)
    print(readme[:20000] + ("\n[... README truncated in this printout; the full file is on Drive]" if len(readme) > 20000 else ""))
    head = pd.read_csv(f"{EXT_RAW}/metadata.csv", nrows=5000, dtype=str, low_memory=False)
    print(f"\nD1 metadata.csv: {len(head.columns)} columns (first 5000 rows inspected)")
    with pd.option_context("display.max_columns", None, "display.width", 250, "display.max_colwidth", 60):
        print(head.head(5).to_string())
    for c in head.columns:
        ex = head[c].dropna().unique()[:6]
        print(f"  {c!r:32s} non-null {head[c].notna().mean():5.1%} | unique (first 5000 rows) {head[c].nunique():6d} | e.g. {list(ex)}")
    return readme, head

def ext_d1_resolve(readme, head):
    cols = list(head.columns)
    unset = [k for k, v in EXT_SCHEMA.items() if not v]
    if unset:
        hint = {"subject_col": r"subj|patient", "label_col": r"rhythm|label|class", "path_col": r"path|file|record|segment"}
        print("\nColumns whose NAME looks related (a hint for you; the notebook does not use it):",
              {k: [c for c in cols if re.search(hint[k], c, re.I)] for k in unset})
        raise StopForUser(f"Set EXT_SCHEMA {unset} in §2 from the README and the metadata.csv columns printed above, then "
                          "run again. If a segment's file location has to be built from several columns, set path_col to "
                          "one of them and EXT_PATH_TEMPLATE accordingly. The notebook does not guess column names.")
    bad = [v for v in EXT_SCHEMA.values() if v not in cols]
    if bad: raise StopForUser(f"EXT_SCHEMA names columns that are not in metadata.csv: {bad}. Columns: {cols}")
    if len(set(EXT_SCHEMA.values())) < 3: raise StopForUser(f"EXT_SCHEMA uses one column twice: {EXT_SCHEMA}")
    print("D1 schema:", EXT_SCHEMA, "| path template:", EXT_PATH_TEMPLATE)
    for k, v in EXT_SCHEMA.items():
        hits = [l.strip() for l in readme.splitlines() if v in l][:3]
        print(f"  README lines mentioning {v!r}:", hits if hits else "NONE (check that this column means what you think)")

def ext_read_metadata():
    sc, lc, pc = EXT_SCHEMA["subject_col"], EXT_SCHEMA["label_col"], EXT_SCHEMA["path_col"]
    code2cls = {str(EXT_LABEL_CODES[c]): c for c in EXT_CLASSES}
    counts, parts = Counter(), []
    for ch in pd.read_csv(f"{EXT_RAW}/metadata.csv", usecols=[sc, lc, pc], dtype=str, chunksize=500_000):
        lab = ch[lc].fillna("<NA>").str.strip()
        counts.update(lab.value_counts().to_dict())
        m = lab.isin(list(code2cls))
        parts.append(pd.DataFrame({"subject": ch.loc[m, sc].str.strip(), "path": ch.loc[m, pc].str.strip(),
                                   "label": lab[m].map(code2cls)}))
    vc = pd.Series(counts).sort_values(ascending=False)
    print(f"D1 label column {lc!r}: {len(vc)} distinct values over {vc.sum()} rows. Top 40:\n" + vc.head(40).to_string())
    missing = [c for c in EXT_CLASSES if str(EXT_LABEL_CODES[c]) not in vc.index]
    if missing:
        raise StopForUser(f"No exact match in {lc!r} for the codes of {missing} (EXT_LABEL_CODES = {EXT_LABEL_CODES}). "
                          "Look up the code of each class in the README and set EXT_LABEL_CODES in §2.")
    for c in EXT_CLASSES:
        code = str(EXT_LABEL_CODES[c])
        compound = [v for v in vc.index if v != code and code in re.split(r"[^A-Za-z0-9]+", str(v))]
        if compound:
            raise StopForUser(f"Values {compound[:8]} contain the code {code!r} together with other text, so the label column "
                              "may hold several rhythms per segment. Decide how to treat them (edit ext_read_metadata) "
                              "and run again.")
    meta = pd.concat(parts, ignore_index=True).dropna()
    meta = meta[(meta.subject != "") & (meta.path != "")].drop_duplicates()
    conflict = meta.groupby("path").agg(nl=("label", "nunique"), ns=("subject", "nunique"))
    if (conflict.nl > 1).any() or (conflict.ns > 1).any():
        bad = conflict[(conflict.nl > 1) | (conflict.ns > 1)].head(5)
        raise StopForUser(f"Some segment paths have several labels or subjects (first: {bad.index.tolist()}). "
                          "path_col does not identify one segment; check EXT_SCHEMA.")
    print("D2 kept rows by class:", meta.label.value_counts().reindex(EXT_CLASSES).to_dict(),
          "| subjects:", meta.subject.nunique())
    return meta

def _round_robin(per_class, m):
    # Take one segment per class in turn until m segments are taken: balanced where possible, then filled up.
    out, ptr = [], {c: 0 for c in per_class}
    while len(out) < m and any(ptr[c] < len(v) for c, v in per_class.items()):
        for c in EXT_CLASSES:
            if c in per_class and ptr[c] < len(per_class[c]) and len(out) < m:
                out.append(per_class[c][ptr[c]]); ptr[c] += 1
    return out

def ext_select(meta, exclude_ids):
    meta = meta.copy()
    meta["sid"] = meta.subject.map(subject_num)
    if exclude_ids is not None:
        if meta.sid.isna().any():
            print(f"⚠ D6: {meta.sid.isna().sum()} rows have a subject value without digits; they can't be checked for overlap")
        ovl = meta.sid.isin(set(exclude_ids))
        print(f"D6 overlap with MIMIC PERform AF: {meta.subject[ovl].nunique()} Ext-PPG subjects ({int(ovl.sum())} "
              f"segments of the kept classes) -> removed before selection: {sorted(meta.subject[ovl].unique().tolist())}")
        meta = meta[~ovl]
    has_af = meta.groupby("subject")["label"].apply(lambda s: bool((s == "AF").any()))
    af_s, non_s = sorted(has_af.index[has_af]), sorted(has_af.index[~has_af])
    rng = np.random.default_rng(EXT_SELECT_SEED)
    n_af = min(len(af_s), N_SUBJECTS // 2); n_non = min(len(non_s), N_SUBJECTS - n_af); n_af = min(len(af_s), N_SUBJECTS - n_non)
    chosen = sorted(rng.choice(af_s, n_af, replace=False).tolist() + rng.choice(non_s, n_non, replace=False).tolist())
    print(f"D3a subjects available: {len(af_s)} with AF, {len(non_s)} without -> picked {n_af} with AF + {n_non} without "
          f"(seed {EXT_SELECT_SEED})")
    rows = []
    for s, g in meta[meta.subject.isin(chosen)].sort_values(["subject", "path"]).groupby("subject", sort=True):
        per = {c: g[g.label == c].path.tolist() for c in EXT_CLASSES if (g.label == c).any()}
        per = {c: [v[i] for i in rng.permutation(len(v))] for c, v in per.items()}
        lab_of = dict(zip(g.path, g.label))
        rows += [(s, p, lab_of[p]) for p in _round_robin(per, MAX_SEG_PER_SUBJECT)]
    sel = pd.DataFrame(rows, columns=["subject", "path", "label"])
    def _rec(r):
        p = EXT_PATH_TEMPLATE.format(path=r.path, subject=r.subject)
        return re.sub(r"\.(hea|dat)$", "", p).lstrip("/")
    sel["record"] = [_rec(r) for r in sel.itertuples()]
    print(f"D3b selected {len(sel)} segments:", sel.label.value_counts().reindex(EXT_CLASSES).to_dict())
    return sel[["subject", "record", "label", "path"]]

def ext_tag(exclude_ids):
    key = json.dumps([N_SUBJECTS, MAX_SEG_PER_SUBJECT, EXT_SELECT_SEED, EXT_CLASSES, EXT_SCHEMA, EXT_PATH_TEMPLATE,
                      {c: str(EXT_LABEL_CODES[c]) for c in EXT_CLASSES}, sorted(exclude_ids or [])], sort_keys=True)
    return f"n{N_SUBJECTS}_m{MAX_SEG_PER_SUBJECT}_s{EXT_SELECT_SEED}_{'mc' if EXT_MULTICLASS else 'bin'}_" + \
           hashlib.sha1(key.encode()).hexdigest()[:8]

def pick_ppg_channel(sig_names):
    low = [str(s).lower().strip() for s in sig_names]
    exact = [i for i, s in enumerate(low) if s in PPG_NAMES]
    part = [i for i, s in enumerate(low) if any(p in s for p in PPG_NAMES)]
    return exact[0] if exact else (part[0] if part else None)

def ext_verify_layout(man, creds, n=3):
    import wfdb
    rows = man.head(n)
    missing = fetch_physionet([r + ".hea" for r in rows.record], EXT_BASE, EXT_RAW, creds, desc="layout check headers")
    if missing:
        raise StopForUser(f"Headers not found on PhysioNet: {missing}. path_col + EXT_PATH_TEMPLATE do not give the file "
                          "location; check the folder layout in the README and set EXT_PATH_TEMPLATE in §2.")
    for r in rows.record:
        hd = wfdb.rdheader(os.path.join(EXT_RAW, r))
        segs = getattr(hd, "seg_name", None)
        print(f"  layout check {r}: fs={hd.fs} sig_len={hd.sig_len} ({hd.sig_len/hd.fs:.1f} s) sig_name={hd.sig_name} "
              f"files={sorted(set(hd.file_name or []))} comments={(hd.comments or [])[:3]}")
        if segs: raise StopForUser(f"{r} is a multi-segment record ({segs[:3]}...); the loader expects one record per segment.")
        if abs(float(hd.fs) - EXT_FS_NATIVE) > 1e-6: raise StopForUser(f"{r}: fs = {hd.fs}, expected {EXT_FS_NATIVE} Hz")
        if pick_ppg_channel(hd.sig_name) is None: raise StopForUser(f"{r}: no PLETH/PPG channel in {hd.sig_name}")
        if not 25 <= hd.sig_len / hd.fs <= 31:
            raise StopForUser(f"{r}: record is {hd.sig_len/hd.fs:.1f} s long, expected a 30 s segment. The metadata may index "
                              "into longer records (start/end columns?). Adapt ext_preprocess before going on.")

def ext_download(man, man_path, creds):
    import wfdb
    miss_h = set(fetch_physionet([r + ".hea" for r in man.record], EXT_BASE, EXT_RAW, creds, n_par=EXT_DOWNLOAD_PARALLEL,
                                 desc="segment headers"))
    sig_files, bad = set(), []
    for r in man.record:
        if r + ".hea" in miss_h: continue
        try:
            hd = wfdb.rdheader(os.path.join(EXT_RAW, r))
            sig_files.update(os.path.join(os.path.dirname(r), f) for f in set(hd.file_name or []))
        except Exception as e:
            bad.append((r, type(e).__name__))
    if bad: print(f"⚠ {len(bad)} unreadable headers, e.g. {bad[:3]}")
    miss_s = set(fetch_physionet(sorted(sig_files), EXT_BASE, EXT_RAW, creds, n_par=EXT_DOWNLOAD_PARALLEL, desc="signal files"))
    def _ok(r):
        if r + ".hea" in miss_h or r in dict(bad): return False
        hd = wfdb.rdheader(os.path.join(EXT_RAW, r))
        return all(os.path.join(os.path.dirname(r), f) not in miss_s for f in set(hd.file_name or []))
    man["downloaded"] = [_ok(r) for r in man.record]
    man.to_csv(man_path, index=False)
    print(f"downloaded {int(man.downloaded.sum())}/{len(man)} selected segments | manifest: {man_path}")
    return man

def clean_segment(sig, fs, min_valid_s=25.0, flat_run_s=1.0):
    """(signal or None, reason). Invalid samples: NaN/inf, or inside a run of identical values lasting >= flat_run_s."""
    sig = np.asarray(sig, np.float64)
    if not np.all(np.isfinite(sig)): return None, "nan"
    if len(sig) == 0 or sig.std() < 1e-6: return None, "flat"
    change = np.flatnonzero(np.diff(sig) != 0) + 1
    starts, ends = np.r_[0, change], np.r_[change, len(sig)]
    flat = int(sum(e - s for s, e in zip(starts, ends) if e - s >= flat_run_s*fs))
    if (len(sig) - flat) / fs < min_valid_s: return None, "<25s valid"
    return sig, "ok"

def ext_preprocess(man, cfg):
    import wfdb
    n_nat = int(round(cfg.win_sec * EXT_FS_NATIVE)); g = math.gcd(EXT_FS_NATIVE, cfg.fs)
    up, down = cfg.fs // g, EXT_FS_NATIVE // g                     # 125 -> 50 Hz: up=2, down=5
    X, y, grp, recs, reasons = [], [], [], [], []
    for i, r in enumerate(man.itertuples()):
        if not r.downloaded: reasons.append("not downloaded"); continue
        try:
            rec = wfdb.rdrecord(os.path.join(EXT_RAW, r.record))
        except Exception as e:
            reasons.append("unreadable"); continue
        ch = pick_ppg_channel(rec.sig_name)
        if ch is None: reasons.append("no PPG channel"); continue
        if abs(float(rec.fs) - EXT_FS_NATIVE) > 1e-6: reasons.append(f"fs {rec.fs}"); continue
        sig, why = clean_segment(rec.p_signal[:n_nat, ch], EXT_FS_NATIVE)
        if sig is None: reasons.append(why); continue
        why = "ok"
        if len(sig) < n_nat: sig = np.pad(sig, (0, n_nat - len(sig)), mode="edge"); why = "ok (edge-padded to 30 s)"
        X.append(resample_poly(sig, up, down).astype(np.float32)); y.append(EXT_CLASSES.index(r.label))
        grp.append(str(r.subject)); recs.append(r.record); reasons.append(why)
        if (i + 1) % 2000 == 0: print(f"\r  preprocessed {i+1}/{len(man)}", end="")
    print()
    man["status"] = reasons
    print("D4 segment status:", pd.Series(reasons).value_counts().to_dict())
    return np.stack(X), np.array(y), np.array(grp), np.array(recs)

def load_mimic3_ext_ppg(cfg, exclude_ids):
    tag = ext_tag(exclude_ids)
    cache = f"{DATA_ROOT}/processed/mimic3_ext_ppg_{cfg.fs}hz_{cfg.win_sec}s{'_mc' if EXT_MULTICLASS else ''}.npz"
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=False)
        if str(d["tag"]) == tag:
            print("loaded cache", cache, "| selection", tag)
            return d["X"], d["y"], d["subject"], d["record"], [str(c) for c in d["class_names"]]
        print(f"cache {cache} was built with another selection ({d['tag']}); rebuilding for {tag}")
    creds = physionet_creds()
    man_path = f"{EXT_RAW}/manifest_{tag}.csv"
    if os.path.exists(man_path):
        man = pd.read_csv(man_path, dtype={"subject": str, "record": str, "label": str, "path": str})
        print("D3 reusing manifest", man_path, f"({len(man)} segments)")
    else:
        readme, head = ext_d1_inspect(creds)
        ext_d1_resolve(readme, head)
        man = ext_select(ext_read_metadata(), exclude_ids)
        ext_verify_layout(man, creds)
        man.to_csv(man_path, index=False); print("D3 manifest written:", man_path)
    man = ext_download(man, man_path, creds)
    X, y, g, recs = ext_preprocess(man, cfg)
    man.to_csv(man_path, index=False)
    np.savez_compressed(cache, X=X, y=y, subject=g, record=recs, class_names=np.array(EXT_CLASSES), tag=np.array(tag))
    print("cached to", cache)
    return X, y, g, recs, EXT_CLASSES
