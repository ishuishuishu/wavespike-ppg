def batches(idx, bs):
    for i in range(0, len(idx), bs): yield idx[i:i+bs]

@torch.no_grad()
def predict_logits(model, Xa, idx, bs=128):
    model.eval(); out = []
    for b in batches(np.asarray(idx), bs):
        lg, _ = model(torch.tensor(Xa[b], device=DEVICE)); out.append(lg.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.cfg.n_classes), np.float32)

def softmax_np(lg):
    e = np.exp(lg - lg.max(1, keepdims=True)); return e/e.sum(1, keepdims=True)

def predict(model, idx, bs=128, Xa=None):
    return softmax_np(predict_logits(model, X if Xa is None else Xa, idx, bs))

def weighted_ce_np(lg, yt, cw):
    # identical to F.cross_entropy(lg, yt, weight=cw) (mean reduction), computed in float64
    lg = lg.astype(np.float64); m = lg.max(1, keepdims=True)
    logp = lg - m - np.log(np.exp(lg - m).sum(1, keepdims=True))
    w = cw[yt]; return float((w*(-logp[np.arange(len(yt)), yt])).sum()/w.sum())

def metrics(y_true, prob):
    K = prob.shape[1]; pred = prob.argmax(1); cm = confusion_matrix(y_true, pred, labels=range(K))
    sens, spec = [], []
    for c in range(K):
        tp = cm[c, c]; fn = cm[c].sum()-tp; fp = cm[:, c].sum()-tp; tn = cm.sum()-tp-fn-fp
        sens.append(tp/max(tp+fn, 1)); spec.append(tn/max(tn+fp, 1))
    try:
        auc = roc_auc_score(y_true, prob[:, 1]) if K == 2 else roc_auc_score(y_true, prob, multi_class="ovr", average="macro", labels=range(K))
    except ValueError: auc = float("nan")
    return {"macro_f1": float(f1_score(y_true, pred, average="macro", labels=range(K), zero_division=0)),
            "acc": float((pred == y_true).mean()), "auroc": float(auc),
            "sens": [float(v) for v in sens], "spec": [float(v) for v in spec], "cm": cm.tolist()}

def _auc(lab, score):
    lab = np.asarray(lab); return float(roc_auc_score(lab, score)) if len(np.unique(lab)) == 2 else float("nan")

def subject_level(p_af, is_af, g):
    """Subject-level AUROC: mean window P(AF) per subject vs 'subject has any AF window'."""
    df = pd.DataFrame({"g": g, "p": p_af, "a": is_af}).groupby("g").agg(p=("p", "mean"), a=("a", "max"))
    return _auc(df.a.values, df.p.values), len(df)

def binary_af_metrics(p_af, is_af, g):
    is_af = np.asarray(is_af).astype(int)
    sa, ns = subject_level(p_af, is_af, g)
    return {"auroc": _auc(is_af, p_af), "macro_f1": float(f1_score(is_af, (p_af >= 0.5).astype(int), average="macro", zero_division=0)),
            "subj_auroc": sa, "n_subjects": ns}

def balanced_subsample(idx, n, rng, ya=None):
    ya = y if ya is None else ya
    if len(idx) <= n: return rng.permutation(idx)
    cls = [idx[ya[idx] == c] for c in np.unique(ya[idx])]; per = n//len(cls)
    return rng.permutation(np.concatenate([rng.choice(c, min(per, len(c)), replace=False) for c in cls]))

def augment_batch(xb, rng, fs):
    # P1.8 training-time augmentation (see §5 text). xb: (B, T) float32, already z-scored.
    B, T = xb.shape
    x = xb*rng.uniform(0.7, 1.3, (B, 1))
    sh = rng.integers(0, T, B); x = x[np.arange(B)[:, None], (np.arange(T)[None, :] - sh[:, None]) % T]
    p = x.var(1, keepdims=True) + 1e-12
    x = x + rng.standard_normal((B, T))*np.sqrt(p/10**(rng.uniform(20, 40, (B, 1))/10))
    t = np.arange(T)[None, :]/fs
    x = x + rng.uniform(0, 0.5, (B, 1))*np.sqrt(p)*np.sin(2*np.pi*rng.uniform(0.05, 0.4, (B, 1))*t + rng.uniform(0, 2*np.pi, (B, 1)))
    return x.astype(np.float32)

def augment_plus_batch(xb, rng, fs):
    # v3.1 training-time augmentation aimed at making the model rely on beat TIMING, not morphology or heart rate:
    # (1) global time-warp by U(0.85, 1.15) (circular, linear interpolation): changes heart rate, keeps relative irregularity;
    # (2) slow multiplicative amplitude modulation (0.05-0.5 Hz, depth up to 0.3). Output is re-standardised per window.
    B, T = xb.shape
    r = rng.uniform(0.85, 1.15, (B, 1)); pos = (np.arange(T)[None, :]*r + rng.uniform(0, T, (B, 1))) % T
    i0 = np.floor(pos).astype(int); fr = pos - i0; i1 = (i0 + 1) % T
    x = xb[np.arange(B)[:, None], i0]*(1 - fr) + xb[np.arange(B)[:, None], i1]*fr
    t = np.arange(T)[None, :]/fs
    x = x*(1 + rng.uniform(0, 0.3, (B, 1))*np.sin(2*np.pi*rng.uniform(0.05, 0.5, (B, 1))*t + rng.uniform(0, 2*np.pi, (B, 1))))
    x = (x - x.mean(1, keepdims=True))/(x.std(1, keepdims=True) + 1e-6)
    return x.astype(np.float32)

def train(cfg, tr_idx, va_idx, verbose=True, Xa=None, ya=None):
    """Returns {"best": model at the lowest val loss (or best val F1 if select_on="val_f1"), "last": model after the last
    epoch run, "hist", "best_epoch", "best_state", "last_state"}."""
    Xa = X if Xa is None else Xa; ya = y if ya is None else ya
    tr_idx, va_idx = np.asarray(tr_idx), np.asarray(va_idx)
    set_seed(cfg.seed); rng = np.random.default_rng(cfg.seed); arng = np.random.default_rng(cfg.seed + 7919)
    model = build_model(cfg).to(DEVICE)
    has_wav = model.swt is not None
    wav = [model.swt.h0] if (has_wav and cfg.learn_wavelet) else []
    rest = [p for n, p in model.named_parameters() if n != "swt.h0"]
    groups_ = [{"params": rest, "lr": cfg.lr}] + ([{"params": wav, "lr": cfg.lr_wavelet, "weight_decay": 0.0}] if wav else [])
    opt = torch.optim.AdamW(groups_, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    cnt = np.bincount(ya[tr_idx], minlength=cfg.n_classes).astype(float)
    cw_np = cnt.sum()/(cfg.n_classes*np.maximum(cnt, 1))
    cw = torch.tensor(cw_np, dtype=torch.float32, device=DEVICE)
    best, best_state, best_ep, hist, bad = -np.inf, None, -1, [], 0
    for ep in range(cfg.epochs):
        model.train(); t0 = time.time(); L, ER, HR = [], [], []
        for b in batches(balanced_subsample(tr_idx, cfg.max_windows_per_epoch, rng, ya), cfg.batch):
            xb = Xa[b]
            if cfg.augment: xb = augment_batch(xb, arng, cfg.fs)
            if cfg.augment_plus: xb = augment_plus_batch(xb, arng, cfg.fs)
            xb = torch.tensor(xb, device=DEVICE); yb = torch.tensor(ya[b], device=DEVICE)
            lg, aux = model(xb)
            loss = F.cross_entropy(lg, yb, weight=cw)
            if has_wav: loss = loss + cfg.lambda_wav*wavelet_loss(model.swt.h0)
            if cfg.arch == "snn": loss = loss + cfg.lambda_rate*F.relu(aux["hid_rate"] - cfg.target_rate)**2
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            L.append(loss.item()); ER.append(float(aux["enc_rate"].detach())); HR.append(float(aux["hid_rate"].detach()))
        sched.step()
        lg_va = predict_logits(model, Xa, va_idx); mv = metrics(ya[va_idx], softmax_np(lg_va))
        vl = weighted_ce_np(lg_va, ya[va_idx], cw_np)
        h = {"epoch": ep, "loss": float(np.mean(L)), "val_loss": vl, "val_f1": mv["macro_f1"], "val_auroc": mv["auroc"],
             "enc_rate": float(np.mean(ER)), "hid_rate": float(np.mean(HR)),
             "wav_loss": float(wavelet_loss(model.swt.h0).item()) if has_wav else 0.0, "sec": time.time()-t0}
        hist.append(h)
        score = -vl if cfg.select_on == "val_loss" else mv["macro_f1"]
        if score > best: best, best_state, best_ep, bad = score, copy.deepcopy(model.state_dict()), ep, 0
        else: bad += 1
        if verbose:
            print(f"ep {ep:2d} loss {h['loss']:.3f} | val loss {vl:.3f} F1 {h['val_f1']:.3f} AUROC {h['val_auroc']:.3f} | "
                  f"enc {h['enc_rate']:.3f} hid {h['hid_rate']:.3f} | wav {h['wav_loss']:.1e} | {h['sec']:.0f}s" + (" *" if best_ep == ep else ""))
            if cfg.arch == "snn" and h["hid_rate"] < 1e-3: print("  ⚠ hidden layer nearly silent")
        if bad >= cfg.patience:
            if verbose: print(f"early stop (no val improvement for {cfg.patience} epochs); best epoch {best_ep}")
            break
    last_state = copy.deepcopy(model.state_dict()); model.eval()
    best_model = build_model(cfg).to(DEVICE); best_model.load_state_dict(best_state); best_model.eval()
    return {"best": best_model, "last": model, "hist": hist, "best_epoch": best_ep,
            "best_state": best_state, "last_state": last_state}

def report(mt, title="TEST", names=None):
    names = names or CLASS_NAMES
    print(f"{title}: macro-F1 {mt['macro_f1']:.3f} | acc {mt['acc']:.3f} | AUROC {mt['auroc']:.3f}")
    for c, n in enumerate(names): print(f"   {n:8s} sens {mt['sens'][c]:.3f}  spec {mt['spec'][c]:.3f}")
    print("   confusion (rows = true):", mt["cm"])
