"""Spiking-wavelet designs P1-P7 from the literature, rebuilt for 1-D PPG (50 Hz, 10 s windows) inside our pipeline.
None of the papers is about PPG; each is reproduced as its WAVELET DESIGN, so that within a comparison only the wavelet
part changes (same data, folds, seeds, recipe, metrics). Every choice a paper leaves open is stated here.

Front-end swaps (plug into our SNN in place of the learnable SWT; HT shrink, ± LIF encoder and recurrent core unchanged):
  P6  Aghabarar et al. 2023 (JAIDM): fixed DWT preprocessing, wavelet family compared -> fixed undecimated DWT with
      pywt filter `cfg.pw_family` (haar, db2, db4, db8, sym4, coif1). Their encoder is constant-current LIF = our encoder.
  P7  Wang et al. 2024 (FWformer, arXiv 2403.18228) DWT-C: Base = a*Bior1.1 + b*Haar + c*Db1 with three learnable scalars
      (init 1/3 each - not given in the paper). NOTE: bior1.1, haar and db1 are the same filter, so this equals one
      learnable gain on the Haar transform. `p7_mix` (our extension) mixes genuinely different bases: db4, sym4, coif1.
  P1  Pedersen, Lindeberg, Gerstoft 2026 (arXiv 2605.09770), DoE: band k = h_exp(mu_k) - h_exp(mu_{k-1}), one leaky
      integrator per scale, geometric mu with c = 2 (their ECG setting), channel normalised by ||psi||_2. mu chosen so the
      cut-offs 1/(2 pi mu) are 12.5, 6.25, 3.13, 1.56, 0.78 Hz = our SWT band edges; channels: x - y1, y1 - y2, ..., y4 - y5, y5.
      Fixed (closed form, as in the paper). `p1_doe_learn` (our extension): the 5 mu are learnable (partly learnable).
      `p1_doe_penc`: also the paper's encoder: fixed threshold 0.1, membrane constant mu_m = dt, no learned gain.
  P2  Pedersen et al. 2026 (arXiv 2602.02020, ICASSP), DoT: cascade of K = 6 leaky integrators with
      mu_1 = c^(1-K) sqrt(tau_max), mu_k = c^(k-K-1) sqrt(c^2-1) sqrt(tau_max); stage-k output = limit-kernel scale k
      (sigma_k = 6.25, 12.5, ... 200 ms with c = 2, tau_max = 0.04 s^2, chosen to cover 0.8-25 Hz like our SWT; the paper's
      default is c = sqrt2 for synthetic signals). Band = difference of adjacent stages; last stage = low-pass. Fixed.
  P4  Zhao et al. 2026 (arXiv 2606.18644, SPWM): fixed Haar DWT + small learnable separable conv on the sub-bands
      (depthwise k = 5 per sub-band, then pointwise mixing across sub-bands; both start as identity). The IDWT pyramid
      merge acts between stacked image layers and has no counterpart in a single 1-D front-end pass; not used.
      Kernel size not given in the paper (5 chosen).
Architecture-level (wavelet inside the network; need a token-mixer core, so all share one small 1-D spiking
Spikformer-style backbone: patch embedding of 4 samples -> 125 tokens, D = 32, 1 block, 4 simulation steps, LIF
threshold 1.0 (0.5 for P3, as stated), spiking MLP, mean-pooled spiking readout):
  stf_ssa   Spikformer spiking self-attention (Q K^T V * 0.125, no softmax) - the baseline all three modify
  P7 stf_fw     FW head: SSA replaced by a parameter-free Haar transform over tokens (plus `stf_fwc`: DWT-C scalars)
  P3 stf_swsa   SpikeWavformer SWSA: SSA output concatenated with a Haar DWT -> conv -> IDWT branch, then linear
  P5 stf_fatm   SWformer FATM: spiking Haar -> block-diagonal weights (K = 2 blocks) -> inverse Haar, with ternary
                {-1,0,1} spikes in the wavelet branch, summed with a depthwise conv (k = 3) and a pointwise conv branch.
  Haar over tokens: 125 tokens are zero-padded to 128 (Haar needs a power of 2). P3 and P5 keep all 128 coefficients
  between the transform and its inverse, so the inverse is exact (fixed 2026-11-23; before, the transform was cut to
  125 coefficients and the last 3 tokens were not reconstructed). The P7 FW head outputs the transform itself, so it
  keeps the first 125 coefficients.
"""
import math
import numpy as np

def _pywt_lowpass(name):
    import pywt
    return np.asarray(pywt.Wavelet(name).dec_lo, np.float64)          # same convention as the notebook's DB4 list

def make_paper_frontends(g):
    import torch, torch.nn as nn, torch.nn.functional as F
    qmf, DB4 = g["qmf_highpass"], np.asarray(g["DB4"])

    def swt(x, h0, levels):                                            # causal a-trous SWT, as LearnableSWT.forward
        a, outs, L = x.unsqueeze(1), [], h0.shape[-1]
        for j in range(levels):
            h1 = qmf(h0); d = 2**j; w = torch.stack([h0, h1]).flip(-1).unsqueeze(1)
            y_ = F.conv1d(F.pad(a, (d*(L-1), 0)), w, dilation=d)/math.sqrt(2.0); a = y_[:, :1]; outs.append(y_[:, 1:2])
        outs.append(a); return torch.cat(outs, 1)

    class Base(nn.Module):
        """h0 = db4 buffer only so the notebook's train() (wavelet_loss(model.swt.h0)) runs unchanged: it is a constant 0."""
        def __init__(s, levels, warm):
            super().__init__(); s.levels, s.warmup = levels, int(warm)
            s.register_buffer("h0", torch.tensor(np.tile(DB4, (levels, 1)), dtype=torch.float32))

    class FixedSWT(Base):                                              # P6
        def __init__(s, levels, family):
            h = _pywt_lowpass(family); super().__init__(levels, sum((2**j)*(len(h)-1) for j in range(levels)))
            s.register_buffer("hf", torch.tensor(h, dtype=torch.float32))
        def forward(s, x): return swt(x, s.hf, s.levels)

    class MixSWT(Base):                                                # P7 DWT-C (and our distinct-bases extension)
        def __init__(s, levels, families):
            hs = [_pywt_lowpass(f) for f in families]
            super().__init__(levels, max(sum((2**j)*(len(h)-1) for j in range(levels)) for h in hs))
            for i, h in enumerate(hs): s.register_buffer(f"hb{i}", torch.tensor(h, dtype=torch.float32))
            s.n = len(hs); s.mix = nn.Parameter(torch.full((len(hs),), 1.0/len(hs)))
        def forward(s, x): return sum(s.mix[i]*swt(x, getattr(s, f"hb{i}"), s.levels) for i in range(s.n))

    def li_fir(mu, dt, n):                                             # truncated causal leaky-integrator kernel, unit DC gain
        t = torch.arange(n, dtype=torch.float32)
        a = torch.exp(-dt/mu); k = (1 - a)*a**t; return k/k.sum()

    def causal(x, k):                                                  # x (B,1,T) * k (n,) causal
        return F.conv1d(F.pad(x, (k.numel()-1, 0)), k.flip(0).view(1, 1, -1))

    class DoE(Base):                                                   # P1
        def __init__(s, fs, learnable):
            dt = 1.0/fs; fc = np.array([12.5, 6.25, 3.125, 1.5625, 0.78125]); mu = 1/(2*np.pi*fc)
            s.n = int(np.ceil(8*mu.max()/dt)); super().__init__(5, s.n); s.dt = dt
            raw = torch.tensor(np.log(mu), dtype=torch.float32)
            if learnable: s.logmu = nn.Parameter(raw)
            else: s.register_buffer("logmu", raw)
        def forward(s, x):
            x1 = x.unsqueeze(1); ys = [x1] + [causal(x1, li_fir(torch.exp(m), s.dt, s.n)) for m in s.logmu]
            return torch.cat([ys[i] - ys[i+1] for i in range(5)] + [ys[5]], 1)   # fixed version is L2-normalised in build()

    class DoT(Base):                                                   # P2
        def __init__(s, fs, K=6, c=2.0, tau_max=0.04):
            dt = 1.0/fs; r = math.sqrt(tau_max)
            mu = [c**(1-K)*r] + [c**(k-K-1)*math.sqrt(c*c-1)*r for k in range(2, K+1)]
            s.mu = mu; s.ns = [max(2, int(np.ceil(8*m/dt))) for m in mu]; super().__init__(K - 1, sum(s.ns)); s.dt = dt
            for i, (m, n) in enumerate(zip(mu, s.ns)): s.register_buffer(f"k{i}", li_fir(torch.tensor(m), dt, n))
        def forward(s, x):
            z = [x.unsqueeze(1)]
            for i in range(len(s.mu)): z.append(causal(z[-1], getattr(s, f"k{i}")))
            z = z[1:]                                                  # stage outputs = scales sigma_1 < ... < sigma_K
            return torch.cat([z[i] - z[i+1] for i in range(len(z)-1)] + [z[-1]], 1)

    class SepSWT(Base):                                                # P4
        def __init__(s, levels, k=5):
            h = _pywt_lowpass("haar"); super().__init__(levels, sum((2**j)*(len(h)-1) for j in range(levels)) + k - 1)
            s.register_buffer("hf", torch.tensor(h, dtype=torch.float32)); C = levels + 1; s.k = k
            dw = torch.zeros(C, 1, k); dw[:, 0, -1] = 1.0; s.dw = nn.Parameter(dw)          # identity at init (causal)
            s.pw = nn.Parameter(torch.eye(C).unsqueeze(-1))
        def forward(s, x):
            b = swt(x, s.hf, s.levels); C = b.shape[1]
            b = F.conv1d(F.pad(b, (s.k - 1, 0)), s.dw, groups=C)
            return F.conv1d(b, s.pw)

    # per-channel L2 normalisation (P1 / P2: x_k / ||psi_k||_2), computed once from the impulse response at init
    def normalised(mod):
        imp = torch.zeros(1, 4096); imp[0, 0] = 1.0
        with torch.no_grad(): r = mod(imp)[0]
        nrm = r.norm(dim=1).clamp_min(1e-6)
        mod.register_buffer("chan_norm", nrm.view(1, -1, 1)); f = mod.forward
        mod.forward = lambda x: f(x)/mod.chan_norm
        return mod

    WF = g["WaveFront"]; orig = WF._build_frontend
    def build(self, cfg):
        pw = getattr(cfg, "pw", None)
        if not pw: return orig(self, cfg)
        fam = pw.split(":")
        if fam[0] == "p6": m = FixedSWT(cfg.levels, fam[1])
        elif fam[0] == "p7": m = MixSWT(cfg.levels, ["bior1.1", "haar", "db1"] if fam[1] == "dwtc" else ["db4", "sym4", "coif1"])
        elif fam[0] == "p1":
            m = DoE(cfg.fs, learnable=(fam[1] == "learn"))
            if fam[1] != "learn": m = normalised(m)
        elif fam[0] == "p2": m = normalised(DoT(cfg.fs))
        elif fam[0] == "p4": m = SepSWT(cfg.levels)
        else: raise ValueError(pw)
        self.swt = m; n_all = m(torch.zeros(1, 1024)).shape[1]
        self.keep = list(range(n_all)); self.band_names = [f"b{c}" for c in self.keep]
        self.C, self.decim = n_all, int(cfg.snn_decim); self.w0 = -(-m.warmup//self.decim)
        self.ht = g["LearnableHT"](self.C, cfg.ht_alpha, cfg.ht_nonneg, cfg.ht_init) if cfg.use_ht else None
        self.enc_gain = nn.Parameter(torch.ones(self.C))
        if getattr(cfg, "pw_paper_encoder", False): self.enc_gain.requires_grad_(False)
    WF._build_frontend = build

    # P1 paper encoder: fixed threshold 0.1 and membrane constant mu_m = dt (beta = e^-1), not learned
    SN = g["WaveSpikeNet"]; orig_init = SN.__init__
    def init(self, cfg):
        orig_init(self, cfg)
        if getattr(cfg, "pw_paper_encoder", False):
            with torch.no_grad():
                b = math.exp(-1.0); self.enc_beta_raw.fill_(math.log(b/(1-b)))
                self.enc_th_pos_raw.fill_(math.log(math.expm1(0.1))); self.enc_th_neg_raw.fill_(math.log(math.expm1(0.1)))
            for p in (self.enc_beta_raw, self.enc_th_pos_raw, self.enc_th_neg_raw): p.requires_grad_(False)
    SN.__init__ = init


def make_token_nets(g):
    """1-D spiking Spikformer-style backbone with interchangeable token mixers (SSA / P7 FW / P3 SWSA / P5 FATM)."""
    import torch, torch.nn as nn, torch.nn.functional as F
    spike = g["spike"]

    class LIF(nn.Module):                     # multi-step LIF over the simulation steps (dim 0), hard reset
        def __init__(s, th=1.0, beta=0.5, slope=4.0, ternary=False): super().__init__(); s.th, s.b, s.sl, s.ter = th, beta, slope, ternary
        def forward(s, X):                    # X: (Tsim, ...)
            v, out = torch.zeros_like(X[0]), []
            for x in X:
                v = s.b*v + x
                if s.ter:                     # SWformer negative spikes: +1 above th, -1 below -th (symmetric Heaviside)
                    z = spike(v - s.th, s.sl) - spike(-v - s.th, s.sl)
                    v = v*(1 - z.abs())
                else:
                    z = spike(v - s.th, s.sl); v = v*(1 - z)
                out.append(z)
            return torch.stack(out)

    def haar_matrix(n):                       # orthonormal Haar matrix, recursive (SWformer eq.), n power of 2
        H = torch.ones(1, 1)
        while H.shape[0] < n:
            m = H.shape[0]
            H = torch.cat([torch.kron(H, torch.tensor([[1., 1.]])), torch.kron(torch.eye(m), torch.tensor([[1., -1.]]))], 0)/math.sqrt(2)
        return H

    class Mixer(nn.Module):
        def __init__(s, kind, D, N, th):
            super().__init__(); s.kind = kind; s.lif = LIF(th)
            Np = 1 << (N - 1).bit_length(); s.Np = Np; s.register_buffer("H", haar_matrix(Np))
            if kind in ("ssa", "swsa"):
                s.q, s.k, s.v = (nn.Linear(D, D) for _ in range(3)); s.lq, s.lk, s.lv = LIF(th), LIF(th), LIF(th)
            if kind == "swsa":                # P3: wavelet branch (Haar DWT over tokens -> conv -> IDWT), concatenated
                s.wconv = nn.Conv1d(D, D, 3, padding=1, groups=D); s.fuse = nn.Linear(2*D, D)
            if kind == "fwc": s.mix = nn.Parameter(torch.full((3,), 1/3))   # P7 DWT-C: a*bior1.1 + b*haar + c*db1 (all Haar)
            if kind == "fatm":                # P5: block-diagonal frequency weights (K = 2), ternary spikes, + conv + pointwise
                s.Kb = 2; s.wb = nn.Parameter(torch.stack([torch.eye(D//2) for _ in range(2)]) + 0.01*torch.randn(2, D//2, D//2))
                s.ter1, s.ter2 = LIF(th, ternary=True), LIF(th, ternary=True)
                s.sl = nn.Conv1d(D, D, 3, padding=1, groups=D); s.cm = nn.Conv1d(D, D, 1)
            s.bn = nn.BatchNorm1d(D)
        def hw(s, X, inv=False):              # Haar over the token axis; X: (..., N, D). Output cut to N (P7 FW head)
            N = X.shape[-2]; Xp = F.pad(X, (0, 0, 0, s.Np - N)); H = s.H.t() if inv else s.H
            return (H @ Xp)[..., :N, :]
        def dwt(s, X):                        # exact Haar over tokens: pad N -> Np, keep all Np coefficients
            return s.H @ F.pad(X, (0, 0, 0, s.Np - X.shape[-2]))
        def idwt(s, W, N):                    # exact inverse of dwt, then back to the N real tokens
            return (s.H.t() @ W)[..., :N, :]
        def forward(s, X):                    # X: (Tsim, B, N, D) spikes
            Ts, B, N, D = X.shape
            if s.kind == "ssa" or s.kind == "swsa":
                Q, K, V = s.lq(s.q(X)), s.lk(s.k(X)), s.lv(s.v(X))
                Y = (Q @ K.transpose(-1, -2)) @ V*0.125
                if s.kind == "swsa":
                    Np = s.Np; W = s.dwt(X)
                    W = s.wconv(W.reshape(Ts*B, Np, D).transpose(1, 2)).transpose(1, 2).reshape(Ts, B, Np, D)
                    Y = s.fuse(torch.cat([Y, s.idwt(W, N)], -1))
            elif s.kind == "fw": Y = s.hw(X)                                   # P7: parameter-free
            elif s.kind == "fwc": Y = s.mix.sum()*s.hw(X)                      # = one gain: the three bases coincide
            elif s.kind == "fatm":
                Wc = s.ter1(s.dwt(X))                                          # spiking Haar (ternary), all Np coefficients
                Wc = torch.cat([Wc[..., i*(D//2):(i+1)*(D//2)] @ s.wb[i] for i in range(2)], -1)
                Y = s.ter2(s.idwt(Wc, N))
                Z = X.reshape(Ts*B, N, D).transpose(1, 2)
                Y = Y + (s.sl(Z) + s.cm(Z)).transpose(1, 2).reshape(Ts, B, N, D)
            Y = s.bn(Y.reshape(-1, N, D).transpose(1, 2)).transpose(1, 2).reshape(Ts, B, N, D)
            return s.lif(Y)

    class TokenNet(nn.Module):
        def __init__(s, cfg):
            super().__init__(); s.cfg = cfg; s.swt = None; s.w0 = 0; D, P = 32, 4; s.P, s.Ts = P, 4
            th = 0.5 if cfg.mixer == "swsa" else 1.0
            s.emb = nn.Conv1d(1, D, P, stride=P); s.bn0 = nn.BatchNorm1d(D); s.l0 = LIF(th)
            N = -(-cfg.fs*cfg.win_sec // P)
            s.mixer = Mixer(cfg.mixer, D, N, th)
            s.mlp1, s.mlp2 = nn.Linear(D, 2*D), nn.Linear(2*D, D); s.lm1, s.lm2 = LIF(th), LIF(th)
            s.out = nn.Linear(D, cfg.n_classes)
        def forward(s, x, record=False):
            B = x.shape[0]; e = s.bn0(s.emb(x.unsqueeze(1))).transpose(1, 2)       # (B, N, D), same input every sim step
            X = s.l0(e.unsqueeze(0).expand(s.Ts, -1, -1, -1).contiguous())
            X = X + s.mixer(X)
            X = X + s.lm2(s.mlp2(s.lm1(s.mlp1(X))))
            rate = X.abs().mean()
            return s.out(X.mean((0, 2))), {"enc_rate": rate, "hid_rate": rate, "enc_rate_ch": None}

    ob = g["build_model"]
    g["build_model"] = lambda cfg: TokenNet(cfg) if cfg.arch == "stf" else ob(cfg)
