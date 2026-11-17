ZENODO_AF = "https://zenodo.org/records/6807403/files/{}?download=1"
MIMIC_PERFORM_ZIPS = [(1, "mimic_perform_af_wfdb.zip"), (0, "mimic_perform_non_af_wfdb.zip")]

def _mimic_perform_dirs():
    raw = f"{DATA_ROOT}/raw/mimic_perform_af"; os.makedirs(raw, exist_ok=True)
    return [(label, name, extract_once(download_once(ZENODO_AF.format(name), f"{raw}/{name}"), f"{raw}/{name[:-4]}"))
            for label, name in MIMIC_PERFORM_ZIPS]

def load_mimic_perform_af(cfg):
    import wfdb
    X, y, g = [], [], []
    for label, name, d in _mimic_perform_dirs():
        heads = sorted(glob.glob(f"{d}/**/*.hea", recursive=True))
        print(f"{name}: {len(heads)} WFDB headers")
        for h in heads:
            rec = wfdb.rdrecord(h[:-4])
            names = [s.lower() for s in rec.sig_name]
            idx = [i for i, s in enumerate(names) if "ppg" in s or "pleth" in s]
            if not idx:
                print("  no PPG channel in", os.path.basename(h), rec.sig_name); continue
            sid = f"{label}_{os.path.basename(h)[:-4]}"
            for w in windows_from_signal(rec.p_signal[:, idx[0]], rec.fs, cfg.fs, cfg.win_sec):
                X.append(w); y.append(label); g.append(sid)
    return np.array(X, np.float32), np.array(y), np.array(g), ["non-AF", "AF"]

# D6: where is the MIMIC-III subject ID of a MIMIC PERform record? We search the record name, the header's record
# line and its comments (the whole .hea text) for MIMIC-III waveform-style IDs ("p000123", as in the matched
# subset folder/record names) or an explicit "subject_id". Whatever is found, or not found, is printed.
MIMIC3_ID_PATTERNS = [r"(?<![A-Za-z0-9])p(\d{6})(?![0-9])", r"subject[ _-]?id\D{0,3}(\d{1,7})"]

def mimic_perform_subject_ids():
    out = {}
    for label, name, d in _mimic_perform_dirs():
        for h in sorted(glob.glob(f"{d}/**/*.hea", recursive=True)):
            sid = f"{label}_{os.path.basename(h)[:-4]}"
            text = os.path.basename(h)[:-4] + "\n" + open(h, encoding="utf-8", errors="replace").read()
            ids = sorted({int(m.group(1)) for pat in MIMIC3_ID_PATTERNS for m in re.finditer(pat, text, re.I)})
            out[sid] = {"ids": ids, "header": text}
    return out

DEEPBEAT_MAX_PER_SUBJECT = 300

def load_deepbeat(cfg):
    # Best-effort converter: the file layout on Synapse was not verified. If it fails, run inventory(raw)
    # and adapt the key names below.
    raw = synapse_sync_once("syn21985690", f"{DATA_ROOT}/raw/deepbeat")
    files = sorted(glob.glob(f"{raw}/**/*.npz", recursive=True))
    if not files: inventory(raw); raise RuntimeError("no .npz files found; see the inventory above")
    pick = lambda d, keys: next((k for k in keys if k in d.files), None)
    X, y, g = [], [], []; rng = np.random.default_rng(0)
    for p in files:
        d = np.load(p, allow_pickle=False)          # never unpickle downloaded files
        ks, kr, kp = pick(d, ["signal", "signals", "X", "ppg"]), pick(d, ["rhythm", "label", "labels", "y"]), pick(d, ["parameters", "subject", "subjects", "ids"])
        if ks is None or kr is None or kp is None:
            inventory(raw); raise RuntimeError(f"{os.path.basename(p)}: keys {d.files}; set the signal/rhythm/subject keys in load_deepbeat")
        sig, rhy, par = d[ks], d[kr], d[kp]
        sig = sig.reshape(len(sig), -1)
        lab = rhy.argmax(1) if rhy.ndim == 2 else rhy.astype(int)
        sub = par[:, 0] if getattr(par, "ndim", 1) == 2 else par            # assumption: first column = subject id
        keep = []                                    # cap windows per subject (500K+ segments won't fit in Colab RAM)
        for s_ in np.unique(sub):
            ii = np.where(sub == s_)[0]; keep.extend(rng.choice(ii, min(len(ii), DEEPBEAT_MAX_PER_SUBJECT), replace=False))
        for i in sorted(keep):
            X.append(resample_poly(sig[i].astype(np.float64), 25, 16)); y.append(int(lab[i])); g.append(str(sub[i]))   # 32 -> 50 Hz
    print("check: DeepBeat label encoding assumed 0 = non-AF, 1 = AF. Verify on the first run.")
    return np.array(X, np.float32), np.array(y), np.array(g), ["non-AF", "AF"]

def load_uconn(syn_id, name):
    def _loader(cfg):
        raw = synapse_sync_once(syn_id, f"{DATA_ROOT}/raw/{name}")
        inventory(raw)
        raise NotImplementedError(f"{name}: downloaded to {raw}. The file layout is not public, so the converter "
                                  "isn't written yet. Share the inventory printed above and I'll write it. "
                                  "Loader code from the dataset authors: github.com/Cassey2016/UMass_Simband_Dataset")
    return _loader

def load_npz_file(cfg):
    d = np.load(NPZ_PATH, allow_pickle=True)
    X = [resample_poly(x.astype(np.float64), cfg.fs//math.gcd(NPZ_FS, cfg.fs), NPZ_FS//math.gcd(NPZ_FS, cfg.fs)) for x in d["X"]]
    names = list(d["class_names"]) if "class_names" in d.files else [str(c) for c in np.unique(d["y"])]
    return np.array(X, np.float32), d["y"].astype(int), d["subject"].astype(str), names
