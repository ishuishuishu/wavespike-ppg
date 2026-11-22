DB4 = [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114,
       -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523]

def inv_softplus(v): return math.log(math.expm1(v))
def logit(p):        return math.log(p/(1-p))

def init_lowpass(name, L):
    if name == "db4": assert L == 8; return np.array(DB4)
    if name == "haar": h = np.zeros(L); h[-2:] = 1/np.sqrt(2); return h
    raise ValueError(name)

def qmf_highpass(h0):
    L = h0.shape[-1]
    sign = torch.tensor([(-1.0)**k for k in range(L)], dtype=h0.dtype, device=h0.device)
    return sign*torch.flip(h0, [-1])

class LearnableSWT(nn.Module):
    def __init__(self, levels, h0_init, learnable=True, per_level=True):
        super().__init__()
        self.levels, self.L, self.per_level = levels, len(h0_init), per_level
        h0 = torch.tensor(np.tile(np.asarray(h0_init, np.float64), (levels if per_level else 1, 1)), dtype=torch.float32)
        if learnable: self.h0 = nn.Parameter(h0)
        else:         self.register_buffer("h0", h0)
        self.warmup = sum((2**j)*(self.L-1) for j in range(levels))

    def lowpass(self, j): return self.h0[j if self.per_level else 0]

    def forward(self, x):                                     # (B, T) -> (B, levels+1, T)
        a, outs = x.unsqueeze(1), []
        for j in range(self.levels):
            h0 = self.lowpass(j); h1 = qmf_highpass(h0); d = 2**j
            w = torch.stack([h0, h1]).flip(-1).unsqueeze(1)   # flip: conv1d is cross-correlation
            y_ = F.conv1d(F.pad(a, (d*(self.L-1), 0)), w, dilation=d) / math.sqrt(2.0)
            a = y_[:, :1]; outs.append(y_[:, 1:2])
        outs.append(a)
        return torch.cat(outs, 1)

def wavelet_loss(H0):
    H0 = H0 if H0.dim() == 2 else H0.unsqueeze(0); L = H0.shape[-1]; tot = 0.0
    for h0 in H0:
        h1 = qmf_highpass(h0)
        ac = F.conv1d(h0.view(1, 1, -1), h0.view(1, 1, -1), padding=L-1).view(-1)
        even = ac[L-1::2]; tgt = torch.zeros_like(even); tgt[0] = 1.0
        tot = tot + ((even-tgt)**2).sum() + h1.sum()**2
    return tot/H0.shape[0]

class LearnableHT(nn.Module):
    # Michau, Frusque & Fink 2021, eq. 5: HT(x) = x·[σ(−α(x+b⁻)) + σ(α(x−b⁺))]; one (b⁺, b⁻) pair per band.
    # nonneg=True (P1.6): b = softplus(raw) >= 0, so the gain at x -> 0 is σ(−αb⁻)+σ(−αb⁺) <= 1 (shrink only).
    def __init__(self, C, alpha=10.0, nonneg=True, init=0.02):
        super().__init__(); self.alpha, self.nonneg = alpha, nonneg
        if nonneg:
            self.b_pos_raw = nn.Parameter(torch.full((C,), inv_softplus(init)))
            self.b_neg_raw = nn.Parameter(torch.full((C,), inv_softplus(init)))
        else:                                                 # unconstrained DeSpaWN version, starts as an identity
            self.b_pos = nn.Parameter(torch.zeros(C)); self.b_neg = nn.Parameter(torch.zeros(C))
    def thresholds(self):
        if self.nonneg: return F.softplus(self.b_pos_raw), F.softplus(self.b_neg_raw)
        return self.b_pos, self.b_neg
    def gain_at_zero(self):
        bp, bn = self.thresholds(); return torch.sigmoid(-self.alpha*bn) + torch.sigmoid(-self.alpha*bp)
    def forward(self, x):                                     # (B, C, T)
        bp, bn = self.thresholds(); bp, bn = bp.view(1, -1, 1), bn.view(1, -1, 1)
        return x*(torch.sigmoid(-self.alpha*(x + bn)) + torch.sigmoid(self.alpha*(x - bp)))

print("db4 wavelet loss (≈0):", float(wavelet_loss(torch.tensor(DB4))))
_ht = LearnableHT(1, 10.0, True, 0.02)
with torch.no_grad(): print("HT (ht_nonneg) at init: b = %.4f, gain at x→0 = %.3f (≤ 1)" % (float(_ht.thresholds()[0]), float(_ht.gain_at_zero())))
