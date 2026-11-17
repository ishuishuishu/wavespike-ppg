# P0.4a baseline features: pulse-interval irregularity (also used for the P1.9 label-noise check)
FEAT_NAMES = ["cv", "rmssd_over_mean", "frac_dI_gt_15pct"]

def pulse_intervals(x, fs):
    # Assumption: a 0.5–8 Hz zero-phase band-pass before find_peaks (baseline wander otherwise breaks the peak picking).
    # Peaks at least 0.27 s apart (<= 222 bpm), prominence >= 0.3 SD of the filtered window.
    xf = sosfiltfilt(butter(2, [0.5, 8.0], btype="band", fs=fs, output="sos"), np.asarray(x, np.float64))
    pk, _ = find_peaks(xf, distance=max(1, int(0.27*fs)), prominence=0.3*xf.std())
    return np.diff(pk)/fs

def irregularity_features(x, fs):
    ii = pulse_intervals(x, fs)
    if len(ii) < 4: return [np.nan]*3
    m = ii.mean(); d = np.diff(ii)
    return [ii.std()/m, np.sqrt(np.mean(d**2))/m, float(np.mean(np.abs(d) > 0.15*m))]

def feature_matrix(Xa, fs, cache=None):
    if cache and os.path.exists(cache): return np.load(cache)["F"]
    Fm = np.array([irregularity_features(x, fs) for x in Xa], np.float64)
    if cache: np.savez(cache, F=Fm)
    return Fm

class IrregLR:
    """Logistic regression on the 3 interval features (NaN -> training median, standardised, balanced class weights)."""
    def __init__(self, K): self.K = K
    def fit(self, Fm, yv):
        self.med = np.nan_to_num(np.nanmedian(Fm, 0), nan=0.0)
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
        self.clf.fit(self._imp(Fm), yv); return self
    def _imp(self, Fm): return np.where(np.isnan(Fm), self.med, Fm)
    def predict_proba(self, Fm):
        P = np.zeros((len(Fm), self.K)); P[:, self.clf.classes_] = self.clf.predict_proba(self._imp(Fm)); return P

t0 = time.time()
FEATS = feature_matrix(X, CFG.fs, f"{DATA_ROOT}/processed/feats_{DATASET}_{DATA_FP[:10]}.npz")
FEATS_E = feature_matrix(XE, CFG.fs, f"{DATA_ROOT}/processed/feats_mimic_perform_af_ext.npz") if XE is not None else None
print(f"irregularity features: {FEATS.shape} ({time.time()-t0:.0f}s) | windows with < 4 intervals: {np.isnan(FEATS[:, 0]).mean():.1%}")

# P1.9 label-noise rule. Heuristic thresholds, not validated: "regular" = CV < 0.06 and < 10 % of successive interval
# changes > 15 %; "irregular" = CV > 0.15 and > 40 % of successive changes > 15 %.
REG_RULE   = {"cv_max": 0.06, "frac_max": 0.10}
IRREG_RULE = {"cv_min": 0.15, "frac_min": 0.40}
def looks_regular(Fm):   return (Fm[:, 0] < REG_RULE["cv_max"]) & (Fm[:, 2] < REG_RULE["frac_max"])
def looks_irregular(Fm): return (Fm[:, 0] > IRREG_RULE["cv_min"]) & (Fm[:, 2] > IRREG_RULE["frac_min"])
SR_IDX = CLASS_NAMES.index("SR") if "SR" in CLASS_NAMES else (0 if AF_IDX is not None else None)
def label_disagrees(idx):
    """True where the feature-based rhythm strongly disagrees with the label (AF but regular, SR/non-AF but irregular)."""
    idx = np.asarray(idx); Fm = FEATS[idx]                # NaN features compare False: never regular or irregular
    if AF_IDX is None: return np.zeros(len(idx), bool)
    with np.errstate(invalid="ignore"):
        return ((y[idx] == AF_IDX) & looks_regular(Fm)) | ((y[idx] == SR_IDX) & looks_irregular(Fm))
