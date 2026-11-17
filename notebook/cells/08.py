# Synthetic generator (smoke test only; heuristic, not a physiological model)
def _beat(n, shape):
    t = np.linspace(0, 1, n, endpoint=False); a1, m1, s1, a2, m2, s2 = shape
    return a1*np.exp(-0.5*((t-m1)/s1)**2) + a2*np.exp(-0.5*((t-m2)/s2)**2)

def synth_window(label, rng, subj, fs=50, sec=30):
    N = fs*sec; out = []; total = 0; hr = subj["hr"]*rng.uniform(0.93, 1.07); rr0 = 60.0/hr
    ect = set(rng.choice(np.arange(3, 3+int(sec/rr0)-4), size=rng.integers(3, 7), replace=False)) if label == 2 else set()
    k = 0; pause = False
    while total < N + fs:
        if label == 0:   rr = rr0*(1 + 0.04*np.sin(2*np.pi*0.25*total/fs) + 0.02*rng.standard_normal())
        elif label == 1: rr = rr0*np.clip(1 + 0.22*rng.standard_normal(), 0.45, 1.8)
        else:
            if k in ect:  rr = rr0*rng.uniform(0.55, 0.75); pause = True
            elif pause:   rr = rr0*rng.uniform(1.25, 1.45); pause = False
            else:         rr = rr0*(1 + 0.02*rng.standard_normal())
        n = max(int(rr*fs), 8); amp = np.clip(rr/rr0, 0.4, 1.4)**(0.8 if label else 0.2)
        out.append(amp*_beat(n, subj["shape"])); total += n; k += 1
    x = np.concatenate(out)[:N]; tt = np.arange(N)/fs
    x = x + 0.3*np.sin(2*np.pi*rng.uniform(0.05, 0.3)*tt + rng.uniform(0, 6.3)) + 0.05*rng.standard_normal(N)
    if rng.random() < 0.3:
        s = rng.integers(0, N-fs*3); x[s:s+fs*2] += rng.normal(0, 0.6, fs*2).cumsum()*0.05
    return x

def load_synthetic(cfg, n_subjects=48, win_per_subject=25, seed=123):
    rng = np.random.default_rng(seed); X, y, g = [], [], []
    for s in range(n_subjects):
        subj = {"hr": rng.uniform(55, 100), "shape": (1.0, rng.uniform(.18, .3), rng.uniform(.06, .1), rng.uniform(.25, .6), rng.uniform(.45, .6), rng.uniform(.08, .14))}
        p = rng.dirichlet([1.0, 1.0, 1.0])
        for _ in range(win_per_subject):
            lab = int(rng.choice(3, p=p)); X.append(synth_window(lab, rng, subj, cfg.fs, cfg.win_sec)); y.append(lab); g.append(str(s))
    return np.array(X, np.float32), np.array(y), np.array(g), ["NSR", "AF", "PAC/PVC"]

# ---- Zenodo 10.5281/zenodo.23597128: long-term wrist PPG (100 Hz, green LED) with ECG-derived beat-by-beat AF annotations ----
# Open, no account; CC BY-NC-SA 4.0 (non-commercial, share-alike). Only the PPG is used as model input; the ECG files only provide labels.
# Files are read from $DATA_ROOT/raw/zenodo_23597128 (download them first; this loader never downloads 60 GB by itself).
ZEN_PPG_FS, ZEN_ECG_FS = 100, 500
ZEN_MAX_PER_CLASS = int(os.environ.get("WAVESPIKE_ZEN_MAX_PER_CLASS", 300))   # windows per subject and class (random, seeded)
ZEN_MIN_BEATS = 12                     # a window needs >= 12 annotated beats to get a label
ZEN_AF_MIN_FRAC = 0.95                 # AF window: >= 95 % of its beats annotated AF; non-AF window: 0 AF beats; mixed windows are skipped

def _zen_text(h5, ds):
    return "".join(chr(int(c)) for c in np.array(ds).ravel())

def _zen_seconds(day, tod):            # ("02", "08:14:41") -> seconds since the first monitoring day, 00:00
    h, m, s_ = (int(v) for v in tod.split(":")); return (int(day) - 1)*86400 + h*3600 + m*60 + s_

def load_zenodo_wrist_af(cfg):
    import h5py
    raw = f"{DATA_ROOT}/raw/zenodo_23597128"; rng = np.random.default_rng(0)
    subs = sorted(f[:3] for f in os.listdir(raw) if f.endswith("_PPG.mat") and os.path.exists(f"{raw}/{f[:3]}_ECG.mat"))
    if not subs: raise StopForUser(f"No <id>_PPG.mat / <id>_ECG.mat pairs in {raw}. Download them from https://zenodo.org/records/23597128 first.")
    n = int(cfg.win_sec*ZEN_PPG_FS); g_ = math.gcd(ZEN_PPG_FS, cfg.fs); up, down = cfg.fs//g_, ZEN_PPG_FS//g_
    X, y, g = [], [], []
    for sid in subs:
        e = h5py.File(f"{raw}/{sid}_ECG.mat"); p = h5py.File(f"{raw}/{sid}_PPG.mat")      # an unreadable/truncated file raises: no silent skipping
        af = np.array(e["AF_annotation"]).ravel(); bt = np.array(e["QRSindex"]).ravel()[1:len(af) + 1]/ZEN_ECG_FS
        e0 = _zen_seconds(_zen_text(e, e["recording_startday"]), _zen_text(e, e["recording_starttime"]))
        refs = p["PPG_GREEN"]; cand = []                                                # (segment, window index, label)
        for k in range(refs.shape[0]):
            ds = p[refs[k, 0]]; s0 = _zen_seconds(_zen_text(p, p[p["recording_startday"][k, 0]]), _zen_text(p, p[p["recording_starttime"][k, 0]])) - e0
            for i in range(ds.shape[-1]//n):
                t0 = s0 + i*cfg.win_sec; lo, hi = np.searchsorted(bt, [t0, t0 + cfg.win_sec]); b = af[lo:hi]
                if len(b) < ZEN_MIN_BEATS or np.isnan(b).any(): continue
                fr = float(b.mean())
                if fr >= ZEN_AF_MIN_FRAC: cand.append((k, i, 1))
                elif fr == 0.0: cand.append((k, i, 0))
        pick = []
        for lab in (0, 1):
            c = [t for t in cand if t[2] == lab]
            if c: pick += [c[j] for j in rng.choice(len(c), min(ZEN_MAX_PER_CLASS, len(c)), replace=False)]
        by_seg = {}
        for k, i, lab in pick: by_seg.setdefault(k, []).append((i, lab))
        kept = Counter()
        for k, items in by_seg.items():
            sig = np.array(p[refs[k, 0]]).ravel()
            for i, lab in items:
                w, why = clean_segment(sig[i*n:(i + 1)*n], ZEN_PPG_FS)
                if w is None: continue
                X.append(resample_poly(w, up, down).astype(np.float32)); y.append(lab); g.append(sid); kept[lab] += 1
        print(f"zenodo_wrist_af subject {sid}: {len(cand)} labelled windows, kept non-AF {kept[0]} / AF {kept[1]}")
    return np.array(X, np.float32), np.array(y), np.array(g), ["non-AF", "AF"]

LOADERS = {"mimic_perform_af": (load_mimic_perform_af, 30), "deepbeat": (load_deepbeat, 25),
           "uconn_simband": (load_uconn("syn25005551", "uconn_simband"), 30),
           "uconn_mimic3": (load_uconn("syn25005553", "uconn_mimic3"), 30),
           "pulsewatch": (load_uconn("syn23565056", "pulsewatch"), 30),
           "npz": (load_npz_file, 30), "synthetic": (load_synthetic, 30),
           "mimic3_ext_ppg": (None, 30), "zenodo_wrist_af": (load_zenodo_wrist_af, 30)}            # handled by load_mimic3_ext_ppg (needs the D6 exclusion list)
