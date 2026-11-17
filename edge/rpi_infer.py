
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

    def classify_window(self, x):
        # x: one window of raw samples at 50 Hz. Returns (class, probs, hidden spike rate, logits).
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
