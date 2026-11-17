class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, slope):
        ctx.save_for_backward(v); ctx.slope = slope; return (v >= 0).to(v.dtype)
    @staticmethod
    def backward(ctx, g):
        (v,) = ctx.saved_tensors; return g/(ctx.slope*v.abs() + 1.0)**2, None
spike = SpikeFn.apply

def readout_weights(bo, T, w0):
    # Exact: mean_{t>=w0} vo[t] with vo[t] = bo*vo[t-1] + z[t]  ==  sum_s z[s]*W[s] / (T-w0)
    s = torch.arange(T, device=bo.device, dtype=bo.dtype).unsqueeze(1)
    tail = (1 - bo**(T - torch.clamp(s, min=w0))) / (1 - bo)
    lead = bo**torch.clamp(w0 - s, min=0)
    return lead*tail                                          # (T, K)

BAND_NAMES = lambda levels: [f"d{j+1}" for j in range(levels)] + [f"a{levels}"]

class WaveFront(nn.Module):
    """Shared front end: SWT -> keep bands -> HT -> per-band gain -> decimation. Parameter names match v2."""
    def _build_frontend(self, cfg):
        if cfg.use_wavelet:
            self.swt = LearnableSWT(cfg.levels, init_lowpass(cfg.wavelet_init, cfg.taps), cfg.learn_wavelet, cfg.per_level)
            n_all, w0, names = cfg.levels + 1, self.swt.warmup, BAND_NAMES(cfg.levels)
        else:
            assert not cfg.drop_bands, "drop_bands needs the wavelet"
            self.swt, n_all, w0, names = None, 1, 0, ["raw"]
        assert all(0 <= b < n_all for b in cfg.drop_bands), f"drop_bands must be in 0..{n_all-1}"
        self.keep = [c for c in range(n_all) if c not in set(cfg.drop_bands)]
        assert self.keep, "all bands dropped"
        self.band_names = [names[c] for c in self.keep]
        self.C, self.decim = len(self.keep), int(cfg.snn_decim)
        assert self.decim >= 1
        self.w0 = -(-w0 // self.decim)                        # warm-up in (decimated) SNN steps: ceil(w0 / decim)
        self.ht = LearnableHT(self.C, cfg.ht_alpha, cfg.ht_nonneg, cfg.ht_init) if (cfg.use_ht and cfg.use_wavelet) else None
        self.enc_gain = nn.Parameter(torch.ones(self.C))

    def frontend(self, x):                                    # (B, T) -> (B, C, ceil(T/decim))
        if self.swt is not None:
            bands = self.swt(x)
            if self.C < bands.shape[1]:                       # slicing + cat: deterministic backward (no index_add)
                bands = torch.cat([bands[:, c:c+1] for c in self.keep], 1)
        else:
            bands = x.unsqueeze(1)
        if self.ht is not None: bands = self.ht(bands)
        bands = bands*self.enc_gain.view(1, -1, 1)
        if self.decim > 1: bands = bands[:, :, ::self.decim]
        return bands

class WaveSpikeNet(WaveFront):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg; self._build_frontend(cfg)
        C, H, K, n = self.C, cfg.hidden, cfg.n_classes, self.decim
        self.enc_beta_raw = nn.Parameter(torch.full((C,), logit(0.9**n)))
        self.enc_th_pos_raw = nn.Parameter(torch.full((C,), inv_softplus(1.0)))
        self.enc_th_neg_raw = nn.Parameter(torch.full((C,), inv_softplus(1.0)))
        self.fc_in = nn.Linear(2*C, H)
        self.fc_rec = nn.Linear(H, H, bias=False); nn.init.orthogonal_(self.fc_rec.weight, gain=0.5)
        if cfg.tau_het:        # heterogeneous time constants (seconds): one SNN step lasts n/fs s, so beta = exp(-(n/fs)/tau)
            taus = np.exp(np.linspace(np.log(cfg.tau_min), np.log(cfg.tau_max), H))
            self.hid_beta_raw = nn.Parameter(torch.tensor([logit(float(np.exp(-(n/cfg.fs)/t))) for t in taus], dtype=torch.float32))
        else:
            self.hid_beta_raw = nn.Parameter(torch.full((H,), logit(0.95**n)))
        self.fc_out = nn.Linear(H, K)
        self.out_beta_raw = nn.Parameter(torch.full((K,), logit(0.95**n)))

    def forward(self, x, record=False):
        s = self.cfg.surr_slope
        bands = self.frontend(x)
        B, C, T = bands.shape
        be = torch.sigmoid(self.enc_beta_raw); thp = F.softplus(self.enc_th_pos_raw); thn = F.softplus(self.enc_th_neg_raw)
        up = x.new_zeros(B, C); un = x.new_zeros(B, C); ZP, ZN = [], []
        for t in range(T):                                    # encoder loop (independent of the hidden layer)
            b = bands[:, :, t]
            up = be*up + b; un = be*un - b
            zp = spike(up - thp, s); zn = spike(un - thn, s)
            up = up - zp*thp; un = un - zn*thn
            ZP.append(zp); ZN.append(zn)
        S_in = torch.cat([torch.stack(ZP, 1), torch.stack(ZN, 1)], 2)      # (B, T, 2C)
        I = self.fc_in(S_in)                                                # one matmul for all t
        bh = torch.sigmoid(self.hid_beta_raw)
        p_drop = self.cfg.spike_dropout if self.training else 0.0
        vh = x.new_zeros(B, self.cfg.hidden); sh_out = torch.zeros_like(vh); SH, SHraw = [], []
        for t in range(T):                                    # recurrent loop
            vh = bh*vh + I[:, t] + self.fc_rec(sh_out)
            sh = spike(vh - 1.0, s); vh = vh - sh
            sh_out = F.dropout(sh, p_drop, training=True) if p_drop > 0 else sh
            SH.append(sh_out); SHraw.append(sh)
        SH = torch.stack(SH, 1)                                             # (B, T, H) (dropped spikes in training)
        SHr = torch.stack(SHraw, 1) if p_drop > 0 else SH
        w0 = self.w0; assert T > w0, f"window ({T} steps) shorter than the warm-up ({w0})"
        Wr = readout_weights(torch.sigmoid(self.out_beta_raw), T, w0)
        logits = (self.fc_out(SH)*Wr.unsqueeze(0)).sum(1)/(T - w0)
        aux = {"enc_rate": S_in.mean(), "hid_rate": SHr.mean(), "enc_rate_ch": S_in.detach().mean((0, 1))}
        if record: aux["S_in"], aux["SH"] = S_in[0].detach().cpu(), SHr[0].detach().cpu()
        return logits, aux

def _core_params(model):
    return sum(p.numel() for n, p in model.named_parameters() if not n.startswith(("swt.", "ht.", "enc_gain")))

def gru_hidden_matched(cfg):
    # Hidden size whose GRU + readout parameter count is closest to the SNN core (everything after the shared front end).
    target = _core_params(WaveSpikeNet(cfg)); C = (cfg.levels + 1 - len(cfg.drop_bands)) if cfg.use_wavelet else 1; K = cfg.n_classes
    return min(range(2, 513), key=lambda h: abs(3*(C*h + h*h + 2*h) + (h + 1)*K - target))

class WaveGRUNet(WaveFront):
    """Non-spiking twin (P0.4b): same front end, GRU of matched parameter count, linear readout of the mean state."""
    def __init__(self, cfg, hidden=None):
        super().__init__(); self.cfg = cfg; self._build_frontend(cfg)
        self.h = hidden or gru_hidden_matched(cfg)
        self.gru = nn.GRU(self.C, self.h, batch_first=True)
        self.fc_out = nn.Linear(self.h, cfg.n_classes)
    def forward(self, x, record=False):
        bands = self.frontend(x)                                            # (B, C, T)
        out, _ = self.gru(bands.transpose(1, 2).contiguous())
        assert out.shape[1] > self.w0
        z = x.new_zeros(())
        return self.fc_out(out[:, self.w0:].mean(1)), {"enc_rate": z, "hid_rate": z, "enc_rate_ch": None}

def build_model(cfg):
    return {"snn": WaveSpikeNet, "gru": WaveGRUNet}[cfg.arch](cfg)

def n_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

m = WaveSpikeNet(CFG).to(DEVICE)
_g = copy.deepcopy(CFG); _g.arch = "gru"; mg = WaveGRUNet(_g)
print("SNN trainable params:", n_params(m), "| core (after front end):", _core_params(m),
      "| wavelet params:", m.swt.h0.numel() if CFG.learn_wavelet else 0, "| HT params:", 2*m.C if m.ht is not None else 0,
      "| warm-up:", m.w0, "steps")
print(f"GRU twin: hidden {mg.h}, trainable params {n_params(mg)}, core {_core_params(mg)}")
with torch.no_grad():
    t0 = time.time(); lg, aux = m(torch.tensor(X[:CFG.batch], device=DEVICE)); dt = time.time()-t0
print(f"forward on {CFG.batch} windows: {dt:.1f}s | logits {tuple(lg.shape)} | enc {aux['enc_rate']:.3f} hid {aux['hid_rate']:.3f}")
