# step11_rpi_infer.py: NumPy-only inference for the Step 11 WaveSpike-PPG biometric model (Raspberry Pi bundle).
# The spiking kernel is the teammate's rpi_infer.WaveSpikeNP (wavespike-ppg commit 325e222, edge/rpi_infer.py),
# reproduced line for line below, plus three things it lacks:
#   (1) the amplitude-fusion head (logits += W_amp @ z(window level, log pulse SD)), as bio_run.make_amp_fusion;
#   (2) decisions from n consecutive windows (mean log-probability, as bio_run.aggregate);
#   (3) verification against a frozen threshold tau (claimed identity accepted if its decision probability >= tau).
# Only numpy is imported. Usage:  python3 step11_rpi_infer.py model.npz [samples.npz]   (self-check)
import sys, json, time
import numpy as np

def _sigmoid(x):
    with np.errstate(over="ignore"):      # exp(-x) -> inf for very negative x gives the correct limit 0; same values
        return 1.0/(1.0 + np.exp(-x))
def _softplus(x): return np.log1p(np.exp(x))

class WaveSpikeBioNP:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False); f = np.float32
        self.cfg = c = json.loads(str(d["config"]))
        self.levels = int(c["levels"]); self.use_wavelet = bool(c.get("use_wavelet", True))
        self.H0 = np.atleast_2d(d["h0"]).astype(f); self.L = self.H0.shape[1]
        self.H1 = np.stack([np.array([(-1.0)**k*h[self.L-1-k] for k in range(self.L)], dtype=f) for h in self.H0])
        n_all = self.levels + 1 if self.use_wavelet else 1
        drop = set(int(b) for b in c.get("drop_bands", []))
        self.keep = [k for k in range(n_all) if k not in drop]
        self.decim = int(c.get("snn_decim", 1))
        self.use_ht = bool(c["use_ht"]) and self.use_wavelet; self.alpha = f(c["ht_alpha"])
        if self.use_ht and c.get("ht_nonneg", False):
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
        self.warmup = -(-w0 // self.decim)
        self.classes = c["class_names"]
        # Step 11 additions
        self.amp = "amp.weight" in d.files
        if self.amp:
            self.W_amp, self.b_amp = d["amp.weight"].astype(f), d["amp.bias"].astype(f)
            self.amp_mu, self.amp_sd = d["amp_mu"].astype(f), d["amp_sd"].astype(f)
        self.tau = float(d["tau"]) if "tau" in d.files else None
        self.n_agg = int(d["n_agg"]) if "n_agg" in d.files else 1

    def _lp(self, j): return (self.H0[j], self.H1[j]) if self.H0.shape[0] > 1 else (self.H0[0], self.H1[0])

    def bands(self, x):
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
        x = np.asarray(x, np.float32); x = (x - x.mean())/(x.std() + 1e-6)
        bands = self.bands(x)
        if self.use_ht:
            bp, bn = self.bp[:, None], self.bn[:, None]
            bands = bands*(_sigmoid(-self.alpha*(bands + bn)) + _sigmoid(self.alpha*(bands - bp)))
        bands = (bands*self.gain[:, None]).astype(np.float32)
        if self.decim > 1: bands = np.ascontiguousarray(bands[:, ::self.decim])
        return bands

    def amp_features(self, x_raw):
        # [window mean, log window SD] of the RAW window, z-scored with the training-fold statistics (bio_run.main)
        x = np.asarray(x_raw, np.float32)
        a = np.array([x.mean(), np.log(x.std() + np.float32(1e-6))], np.float32)
        return ((a - self.amp_mu)/self.amp_sd).astype(np.float32)

    def window_logits(self, x_raw, stats=None):
        """One 10 s window (raw samples at 50 Hz) -> logits. stats (optional dict) collects spike counts and the closest
        hidden membrane-to-threshold distance (diagnostics only; does not change the result)."""
        bands = self.encoder_input(x_raw)
        C, T = bands.shape; H = self.W_rec.shape[0]; K = self.W_out.shape[0]
        up = np.zeros(C, np.float32); un = np.zeros(C, np.float32)
        vh = np.zeros(H, np.float32); sh = np.zeros(H, np.float32); vo = np.zeros(K, np.float32)
        acc = np.zeros(K, np.float32); margin = np.inf; n_in = 0; n_hid = 0
        for t in range(T):
            b = bands[:, t]
            up = self.be*up + b; un = self.be*un - b
            zp = (up - self.thp >= 0).astype(np.float32); zn = (un - self.thn >= 0).astype(np.float32)
            if stats is not None: margin = min(margin, float(np.abs(up - self.thp).min()), float(np.abs(un - self.thn).min()))
            up -= zp*self.thp; un -= zn*self.thn
            act = np.nonzero(np.concatenate([zp, zn]))[0]
            vh = self.bh*vh + self.W_in[:, act].sum(1) + self.b_in + self.W_rec @ sh
            if stats is not None: margin = min(margin, float(np.abs(vh - 1.0).min()))
            sh = (vh - 1.0 >= 0).astype(np.float32); vh -= sh
            vo = self.bo*vo + self.W_out @ sh + self.b_out
            if t >= self.warmup: acc += vo
            n_in += len(act); n_hid += int(sh.sum())
        logits = acc/(T - self.warmup)
        if self.amp: logits = logits + self.W_amp @ self.amp_features(x_raw) + self.b_amp
        if stats is not None:
            stats["margin"] = margin; stats["in_spikes"] = stats.get("in_spikes", 0) + n_in
            stats["hid_spikes"] = stats.get("hid_spikes", 0) + n_hid; stats["steps"] = stats.get("steps", 0) + T
        return logits.astype(np.float32)

    @staticmethod
    def softmax(lg):
        e = np.exp(lg - lg.max()); return e/e.sum()

    def decide(self, windows_raw):
        """n consecutive windows -> (predicted identity, decision probabilities). Mean log-probability, as bio_run.aggregate."""
        lp = np.mean([np.log(np.clip(self.softmax(self.window_logits(w).astype(np.float64)), 1e-12, 1)) for w in windows_raw], 0)
        p = np.exp(lp - lp.max()); p /= p.sum()
        return self.classes[int(p.argmax())], p

    def verify(self, windows_raw, claimed_index):
        """Accept the claimed identity if its decision probability >= tau (tau frozen on validation at 2 % FAR)."""
        _, p = self.decide(windows_raw); return bool(p[claimed_index] >= self.tau), float(p[claimed_index])

if __name__ == "__main__":
    net = WaveSpikeBioNP(sys.argv[1] if len(sys.argv) > 1 else "snn_bio_model.npz")
    if len(sys.argv) > 2:      # self-check against the reference decisions shipped in the bundle
        s = np.load(sys.argv[2]); W, ref = s["windows"], s["ref_decision_probs"]; n = net.n_agg
        t0 = time.perf_counter(); P = np.stack([net.decide(W[i*n:(i+1)*n])[1] for i in range(len(ref))]); dt = time.perf_counter() - t0
        agree = float((P.argmax(1) == ref.argmax(1)).mean())
        print(f"self-check: {len(ref)} decisions of {n} windows | argmax agreement {agree:.4f} | max |dp| {np.abs(P - ref).max():.2e} | "
              f"{1e3*dt/len(ref):.1f} ms per decision on this machine")
        sys.exit(0 if agree == 1.0 else 1)
    t = np.arange(int(net.cfg["fs"]*net.cfg["win_sec"]))/net.cfg["fs"]; x = np.sin(2*np.pi*1.2*t) + 0.3*np.sin(2*np.pi*2.4*t)
    t0 = time.perf_counter(); lg = net.window_logits(x); dt = time.perf_counter() - t0
    print(f"synthetic window -> {net.classes[int(lg.argmax())]} | {dt*1e3:.1f} ms per window")
