# §2 Configuration
# OPEN ACCESS ONLY: this project uses only datasets that download without an account, application or data use agreement.
OPEN_ACCESS_ONLY = True
OPEN_DATASETS = {"mimic_perform_af": "MIMIC PERform AF (Zenodo, CC BY 4.0)",
                 "synthetic": "generated locally", "npz": "your own file",
                 "zenodo_wrist_af": "Long-term wrist PPG + ECG-annotated AF, Zenodo 10.5281/zenodo.23597128 (open, no account; CC BY-NC-SA 4.0: non-commercial, share-alike)"}
NON_OPEN = {"mimic3_ext_ppg": "PhysioNet credentialed access (CITI training + DUA)",
            "deepbeat": "Synapse account + access token", "uconn_simband": "Synapse + approval by the data owner",
            "uconn_mimic3": "Synapse + approval by the data owner", "pulsewatch": "Synapse, not yet public"}
DATASET   = os.environ.get("WAVESPIKE_DATASET", "mimic_perform_af")   # | "zenodo_wrist_af" | "synthetic" | "npz"   (non-open, blocked while OPEN_ACCESS_ONLY: see NON_OPEN)
USE_DRIVE = True                 # cache downloads + processed data + results on Google Drive
NPZ_PATH, NPZ_FS = "/content/drive/MyDrive/ppg_windows.npz", 50     # only for DATASET="npz" (keys X, y, subject)
SYNTH_KW = {}                    # extra arguments for load_synthetic (DATASET="synthetic" only)

# ---- MIMIC-III-Ext-PPG (D1–D6): credentialed, NOT used while OPEN_ACCESS_ONLY = True (kept for later) ----
N_SUBJECTS          = 600        # D3a: subjects picked before any segment is downloaded (about half with AF)
MAX_SEG_PER_SUBJECT = 40         # D3b: segments per subject, balanced AF/SR where possible
EXT_SELECT_SEED     = 0
EXT_MULTICLASS      = False      # D2: True -> {SR, AF, STACH, SBRAD}; default binary AF vs SR
# Fill these in from the D1 printout (first run stops and prints the README + metadata.csv columns).
# Never guessed: the notebook stops until all three are set and exist in metadata.csv.
EXT_SCHEMA = {"subject_col": None,   # MIMIC-III subject ID of each segment
              "label_col":   None,   # rhythm label of each segment
              "path_col":    None}   # location of each segment's WFDB record
EXT_PATH_TEMPLATE = "{path}"     # record path relative to the project root, extension stripped; fields {path}, {subject}
EXT_LABEL_CODES = {"SR": "SR", "AF": "AF", "STACH": "STACH", "SBRAD": "SBRAD"}   # value in label_col for each class;
                                 # checked for an EXACT match at runtime (stops otherwise)
EXT_DOWNLOAD_PARALLEL = 8        # parallel wget processes
EXTERNAL_DATASET = "mimic_perform_af" if DATASET == "mimic3_ext_ppg" else None    # D6 external test set
RUN_CINC2015 = True              # D7 open-access sanity task (not AF detection); CHANGED: on by default
LABEL_CLEAN_TRAIN = False        # P1.9: drop label/feature disagreements from TRAIN+VAL only; reported as its own variant

# ---- repeated CV (P0.5) and session budget ----
CV_SEEDS    = [0, 1, 2]          # fold seeds
TRAIN_SEEDS = [0, 1]             # training seeds
MAX_SESSION_H = float(os.environ.get("WAVESPIKE_MAX_H", 3.6))              # stop launching new ablation runs after this many hours (Colab T4 ~4 h); rerun to resume

@dataclass
class Config:
    fs: int = 50
    win_sec: int = 30            # overwritten per dataset (DeepBeat: 25 s)
    levels: int = 5              # bands ≈ 12.5-25, 6.3-12.5, 3.1-6.3, 1.6-3.1, 0.8-1.6 Hz + approx <0.8 Hz
    taps: int = 8
    wavelet_init: str = "db4"
    learn_wavelet: bool = True
    per_level: bool = True       # DeSpaWN: one learned filter per level
    use_wavelet: bool = True     # NEW. False = raw signal straight into the encoder (baseline P0.4c)
    use_ht: bool = True          # DeSpaWN: learnable hard-threshold denoising
    ht_alpha: float = 10.0       # DeSpaWN uses α = 10
    ht_nonneg: bool = True       # NEW, CHANGES v2 behaviour: b = softplus(raw) >= 0 (P1.6). False = unconstrained DeSpaWN
    ht_init: float = 0.02        # NEW. initial b when ht_nonneg (raw = inv_softplus(0.02)); the unconstrained HT starts at 0 as in v2
    drop_bands: list = field(default_factory=list)   # NEW (P1.7). band indices to drop: 0 = d1 ... levels-1 = d5, levels = approx
    snn_decim: int = 1           # NEW (P1.7). keep every n-th sample of the kept bands before the encoder
    arch: str = "snn"            # NEW. "snn" | "gru" (non-spiking twin, P0.4b)
    spike_dropout: float = 0.0   # NEW (P1.8). dropout on hidden spikes, training only
    augment: bool = False        # NEW (P1.8). amplitude, circular shift, noise, baseline-wander augmentation
    augment_plus: bool = False   # NEW (v3.1). time-warp (x0.85-1.15), slow amplitude modulation; training only; use together with augment
    tau_het: bool = False        # NEW (v3.1). hidden-layer leaks initialised log-uniformly over tau_min..tau_max seconds (default: all 0.95 per 50 Hz step = 0.39 s)
    tau_min: float = 0.2
    tau_max: float = 4.0
    hidden: int = 64
    n_classes: int = 2           # overwritten by the dataset
    surr_slope: float = 10.0
    epochs: int = 30
    patience: int = 8            # early stopping patience
    select_on: str = "val_loss"  # CHANGED (v2: val macro-F1). "val_loss" (class-weighted CE) | "val_f1"
    batch: int = 64
    max_windows_per_epoch: int = 4000   # class-balanced subsample per epoch (large datasets)
    lr: float = 2e-3
    lr_wavelet: float = 5e-4
    lambda_wav: float = 1.0
    lambda_rate: float = 1.0
    target_rate: float = 0.10
    n_folds: int = 5
    seed: int = 0

CFG = Config()
for _k, _v in json.loads(os.environ.get("WAVESPIKE_CFG", "{}")).items():     # v3.1: optional config overrides (unset = defaults)
    setattr(CFG, _k, _v); print("CFG override:", _k, "=", _v)
if SMOKE:                        # local CPU test: tiny everything (not used in Colab)
    CFG.epochs, CFG.patience, CFG.max_windows_per_epoch, CFG.batch, CFG.n_folds = 2, 1, 64, 32, 3
    CV_SEEDS, TRAIN_SEEDS = [0], [0]
    for _k, _v in json.loads(os.environ.get("WAVESPIKE_OVERRIDES", "{}")).items():
        globals()[_k] = _v
        print("SMOKE override:", _k, "=", _v)
    EXTERNAL_DATASET = "mimic_perform_af" if DATASET == "mimic3_ext_ppg" else None

if USE_DRIVE and IN_COLAB:
    from google.colab import drive; drive.mount("/content/drive")
    DATA_ROOT = "/content/drive/MyDrive/wavespike_data"
else:
    DATA_ROOT = os.environ.get("WAVESPIKE_DATA_ROOT", "/content/wavespike_data")
if OPEN_ACCESS_ONLY and DATASET not in OPEN_DATASETS:
    raise StopForUser(f"DATASET = {DATASET!r} is not open access ({NON_OPEN.get(DATASET, 'unknown access')}). This project "
                      f"uses open data only: pick one of {list(OPEN_DATASETS)}, or set OPEN_ACCESS_ONLY = False.")
for sub in ("raw", "processed", "results"): os.makedirs(f"{DATA_ROOT}/{sub}", exist_ok=True)
print("data root:", DATA_ROOT, "| dataset:", DATASET, "| external:", EXTERNAL_DATASET)
