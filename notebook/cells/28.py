RPI_SRC = r'''
# rpi_infer.py: NumPy-only inference for WaveSpike-PPG v3 (exported from the Colab notebook).
# Usage:  python3 rpi_infer.py wavespike_rpi.npz        (self-test on a synthetic pulse)
# Replace read_window_from_sensor() with your sensor driver. The window length must match training (cfg win_sec at 50 Hz).
# Supports the v3 flags ht_nonneg, drop_bands, snn_decim and use_wavelet; files exported by v2 still load (v2 defaults).
import sys, time, json, numpy as np

def _sigmoid(x): return 1.0/(1.0+np.exp(-x))
def _softplus(x): return np.log1p(np.exp(x))

class WaveSpikeNP:
    def __init__(self, path):
        d = np.load(path); f = np.float32
        self.cfg = c = json.loads(str(d["config"]))
        if c.get("arch", "snn") != "snn": raise ValueError("only the spiking model can be exported")
        self.levels = int(c["levels"]); self.use_wavelet = bool(c.get("use_wavelet", True))
        self.H0 = np.atleast_2d(d["h0"]).astype(f); self.L = self.H0.shape[1]
        self.H1 = np.stack([np.array([(-1.0)**k*h[self.L-1-k] for k in range(self.L)], dtype=f) for h in self.H0])
        n_all = self.levels + 1 if self.use_wavelet else 1
        drop = set(int(b) for b in c.get("drop_bands", []))
        self.keep = [k for k in range(n_all) if k not in drop]          # 0 = d1 ... levels-1 = d<levels>, levels = approx
        self.decim = int(c.get("snn_decim", 1))
        self.use_ht = bool(c["use_ht"]) and self.use_wavelet; self.alpha = f(c["ht_alpha"])
        if self.use_ht and c.get("ht_nonneg", False):                  # b = softplus(raw) >= 0: shrink-only threshold
            self.bp, self.bn = _softplus(d["ht_b_pos_raw"].astype(f)).astype(f), _softplus(d["ht_b_neg_raw"].astype(f)).astype(f)
        elif self.use_ht:
            self.bp, self.bn = d["ht_b_pos"].astype(f), d["ht_b_neg"].astype(f)
        self.gain = d["enc_gain"].astype(f); self.be = _sigmoid(d["enc_beta_raw"]).astype(f)
        self.thp = _softplus(d["enc_th_pos_raw"]).astype(f); self.thn = _softplus(d["enc_th_neg_raw"]).astype(f)
        self.W_in, self.b_in = d["fc_in.weight"].astype(f), d["fc_in.bias"].astype(f)
        self.W_rec = d["fc_rec.weight"].astype(f); self.bh = _sigmoid(d["hid_beta_raw"]).astype(f)
        self.W_out, self.b_out = d["fc_out.weight"].astype(f), d["fc_out.bias"].astype(f)
        self.bo = _sigmoid(d["out_beta_raw"]).astype(f)
        w0 = sum((2**j)*(self.L-1) for j in range(self.levels)) if self.use_wavelet else 0
        self.warmup = -(-w0 // self.decim)                              # in SNN steps (after decimation)
        self.classes = c["class_names"]

    def _lp(self, j): return (self.H0[j], self.H1[j]) if self.H0.shape[0] > 1 else (self.H0[0], self.H1[0])

    def bands(self, x):
        # causal stationary wavelet transform: y[t] = sum_k h[k] a[t-d*k] / sqrt(2), per-level filters.
        # Only the kept detail bands are computed (the low-pass chain is always needed).
        a = x.astype(np.float32)
        if not self.use_wavelet: return a[None, :]
        T = len(a); out = {}; s = np.float32(1/np.sqrt(2))
        for j in range(self.levels):
            h0, h1 = self._lp(j); d = 2**j; ap = np.concatenate([np.zeros(d*(self.L-1), np.float32), a])
            lo = np.zeros(T, np.float32); hi = np.zeros(T, np.float32) if j in self.keep else None
            for k in range(self.L):
                seg = ap[d*(self.L-1-k): d*(self.L-1-k)+T]; lo += h0[k]*seg
                if hi is not None: hi += h1[k]*seg
            if hi is not None: out[j] = hi*s
            a = lo*s
        out[self.levels] = a
        return np.stack([out[k] for k in self.keep])

    def encoder_input(self, x):
        # normalise -> SWT (kept bands) -> HT -> gain -> decimation; returns (C, T // decim) float32
        x = np.asarray(x, np.float32); x = (x - x.mean())/(x.std() + 1e-6)
        bands = self.bands(x)
        if self.use_ht:
            bp, bn = self.bp[:, None], self.bn[:, None]
            bands = bands*(_sigmoid(-self.alpha*(bands + bn)) + _sigmoid(self.alpha*(bands - bp)))
        bands = (bands*self.gain[:, None]).astype(np.float32)
        if self.decim > 1: bands = np.ascontiguousarray(bands[:, ::self.decim])   # all kept bands < fs/(2*decim)
        return bands

    def classify_window(self, x, stats=None):
        # x: one window of raw samples at 50 Hz. Returns (class, probs, hidden spike rate, logits).
        # stats: optional dict; if given, spike counts are accumulated in it (benchmarking only, does not change the result).
        bands = self.encoder_input(x)
        C, T = bands.shape; H = self.W_rec.shape[0]; K = self.W_out.shape[0]
        up = np.zeros(C, np.float32); un = np.zeros(C, np.float32)
        vh = np.zeros(H, np.float32); sh = np.zeros(H, np.float32); vo = np.zeros(K, np.float32)
        acc = np.zeros(K, np.float32); nspk = 0.0
        for t in range(T):
            b = bands[:, t]
            up = self.be*up + b; un = self.be*un - b
            zp = (up - self.thp >= 0).astype(np.float32); zn = (un - self.thn >= 0).astype(np.float32)
            up -= zp*self.thp; un -= zn*self.thn
            act = np.nonzero(np.concatenate([zp, zn]))[0]       # event-driven: only active input columns
            vh = self.bh*vh + self.W_in[:, act].sum(1) + self.b_in + self.W_rec @ sh
            sh = (vh - 1.0 >= 0).astype(np.float32); vh -= sh
            if stats is not None:
                stats["steps"] = stats.get("steps", 0) + 1; stats["in_spikes"] = stats.get("in_spikes", 0) + len(act)
                stats["hid_spikes"] = stats.get("hid_spikes", 0) + int(sh.sum())
            vo = self.bo*vo + self.W_out @ sh + self.b_out
            if t >= self.warmup: acc += vo
            nspk += sh.mean()
        logits = acc/(T - self.warmup); p = np.exp(logits - logits.max()); p /= p.sum()
        return self.classes[int(p.argmax())], p, nspk/T, logits

def read_window_from_sensor(n, fs=50):
    # Replace with your sensor driver (e.g. a MAX3010x over I2C); must return n samples at fs Hz.
    t = np.arange(n)/fs; return np.sin(2*np.pi*1.2*t) + 0.3*np.sin(2*np.pi*2.4*t)

if __name__ == "__main__":
    net = WaveSpikeNP(sys.argv[1] if len(sys.argv) > 1 else "wavespike_rpi.npz")
    n = int(net.cfg["fs"]*net.cfg["win_sec"]); x = read_window_from_sensor(n)
    t0 = time.perf_counter(); cls, p, rate, _ = net.classify_window(x); dt = time.perf_counter() - t0
    print(f"prediction {cls} | probs {np.round(p, 3)} | hidden rate {rate:.3f} | {dt*1e3:.1f} ms per {net.cfg['win_sec']} s window")
'''
with open("rpi_infer.py", "w") as f: f.write(RPI_SRC)
import rpi_infer

def export_rpi(model, cfg, path):
    assert cfg.arch == "snn", "only the SNN can be exported"
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}; C = model.C
    exp = {"h0": sd.get("swt.h0", np.zeros((1, cfg.taps), np.float32)),
           "enc_gain": sd["enc_gain"], "enc_beta_raw": sd["enc_beta_raw"],
           "enc_th_pos_raw": sd["enc_th_pos_raw"], "enc_th_neg_raw": sd["enc_th_neg_raw"],
           "fc_in.weight": sd["fc_in.weight"], "fc_in.bias": sd["fc_in.bias"], "fc_rec.weight": sd["fc_rec.weight"],
           "hid_beta_raw": sd["hid_beta_raw"], "fc_out.weight": sd["fc_out.weight"], "fc_out.bias": sd["fc_out.bias"],
           "out_beta_raw": sd["out_beta_raw"],
           "config": json.dumps({**asdict(cfg), "class_names": CLASS_NAMES, "dataset": DATASET})}
    if model.ht is not None and cfg.ht_nonneg: exp["ht_b_pos_raw"], exp["ht_b_neg_raw"] = sd["ht.b_pos_raw"], sd["ht.b_neg_raw"]
    elif model.ht is not None:                 exp["ht_b_pos"], exp["ht_b_neg"] = sd["ht.b_pos"], sd["ht.b_neg"]
    np.savez(path, **exp)

def equivalence_check(model, cfg, idx, path, n_lat=5):
    export_rpi(model, cfg, path); importlib.reload(rpi_infer); net = rpi_infer.WaveSpikeNP(path)
    lg_t = predict_logits(model, X, idx)
    with torch.no_grad():
        xn = torch.tensor(normalize_windows(X[idx[:8]]), device=DEVICE)
        fe_t = model.frontend(xn).cpu().numpy()
    fe_n = np.stack([net.encoder_input(X[i]) for i in idx[:8]])
    assert fe_t.shape == fe_n.shape, (fe_t.shape, fe_n.shape)
    res = [net.classify_window(X[i]) for i in idx]; lg_n = np.stack([r[3] for r in res])
    t0 = time.perf_counter(); [net.classify_window(X[i]) for i in idx[:n_lat]]; ms = (time.perf_counter() - t0)/n_lat*1e3
    return {"agree": float((lg_t.argmax(1) == lg_n.argmax(1)).mean()), "max_dlogit": float(np.abs(lg_t - lg_n).max()),
            "max_dband": float(np.abs(fe_t - fe_n).max()),
            "ms_per_window": ms, "hid_rate": float(np.mean([r[2] for r in res])), "steps": net.cfg["fs"]*net.cfg["win_sec"]//net.decim}

# 1) the §6 model -> wavespike_rpi.npz (the file for the Pi)
r_main = equivalence_check(model, c0, te[:50], "wavespike_rpi.npz")
if USE_DRIVE:
    shutil.copy("wavespike_rpi.npz", f"{DATA_ROOT}/results/wavespike_rpi_{DATASET}.npz"); shutil.copy("rpi_infer.py", f"{DATA_ROOT}/results/rpi_infer.py")
print("saved wavespike_rpi.npz (%.1f kB) + rpi_infer.py | §6 model: max |logit diff| = %.2e | agreement = %.0f%% | %.0f ms/window"
      % (os.path.getsize("wavespike_rpi.npz")/1024, r_main["max_dlogit"], 100*r_main["agree"], r_main["ms_per_window"]))

# 2) every exportable variant (v3 flags): use the ablation checkpoint (fold seed 0, train seed 0, fold 0) if it exists,
#    otherwise a 1-epoch model, so that spikes actually occur in the check
EQUIV_VARIANTS = ["full", "ht_despawn", "lean", "snn_raw", "h32", "spkdrop", "fixed_wavelet", "shared_filter", "no_ht"]
rows = []
for v in EQUIV_VARIANTS:
    c = variant_config(v, TS0); ck = f"{CKPT_DIR}/{v}_cv{CS0}_ts{TS0}_fold{K0}_best.pt"
    if os.path.exists(ck):
        mdl = build_model(c).to(DEVICE); mdl.load_state_dict(torch.load(ck, map_location=DEVICE)); src = "ablation ckpt"
    else:
        cq = copy.deepcopy(c); cq.epochs = 1; cq.max_windows_per_epoch = min(256, CFG.max_windows_per_epoch)
        mdl = train(cq, tr, va[:128], verbose=False)["last"]; src = "1-epoch model"
    r = equivalence_check(mdl, c, te[:50], f"_equiv_{v}.npz"); os.remove(f"_equiv_{v}.npz")
    rows.append({"variant": v, "weights": src, "agreement": f"{100*r['agree']:.0f}%", "max|Δlogit|": f"{r['max_dlogit']:.1e}", "max|Δband|": f"{r['max_dband']:.1e}",
                 "SNN steps": r["steps"], "ms/window": round(r["ms_per_window"], 1), "hidden rate": round(r["hid_rate"], 3)})
EQ = pd.DataFrame(rows); print(EQ.to_string(index=False))
lat = dict(zip(EQ.variant, EQ["ms/window"]))
print(f"P1.7 latency on THIS machine: full {lat['full']:.0f} ms -> lean (drop d1,d2 + decim 2) {lat['lean']:.0f} ms per window "
      f"({100*(lat['lean']/lat['full'] - 1):+.0f}%). Measure again on the Pi.")
assert all(r_["agreement"] == "100%" for r_ in rows) and r_main["agree"] == 1.0, "PyTorch vs NumPy predictions disagree"
assert max(float(r_["max|Δband|"]) for r_ in rows) < 1e-3, "front end (SWT/HT/drop/decim) differs between PyTorch and NumPy"
print("§10 equivalence check passed: 100% prediction agreement for every exported configuration.")
