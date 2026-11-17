def band_responses(H0, per_level, levels, fs, n=4096):
    sw = LearnableSWT(levels, np.asarray(H0)[0], learnable=False, per_level=per_level)
    sw.h0 = torch.tensor(np.asarray(H0), dtype=torch.float32)
    imp = torch.zeros(1, n); imp[0, 0] = 1.0
    with torch.no_grad(): r = sw(imp)[0].numpy()
    return np.fft.rfftfreq(n, 1/fs), np.abs(np.fft.rfft(r, axis=1))

H_init = np.tile(init_lowpass(CFG.wavelet_init, CFG.taps), (CFG.levels if CFG.per_level else 1, 1))
H_learn = model.swt.h0.detach().cpu().numpy()
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
for j, h in enumerate(H_learn): ax[0].plot(h, "o-", ms=3, label=f"level {j+1}" if CFG.per_level else "shared")
ax[0].plot(H_init[0], "k--", lw=1, label="db4 init"); ax[0].legend(fontsize=7); ax[0].set_title("low-pass taps per level")
for H, ls in [(H_init, ":"), (H_learn, "-")]:
    f, R = band_responses(H, CFG.per_level, CFG.levels, CFG.fs)
    for j in range(R.shape[0]): ax[1].plot(f, R[j], ls, color=f"C{j}", label=(f"d{j+1}" if j < CFG.levels else f"a{CFG.levels}") if ls == "-" else None)
ax[1].axvspan(0.7, 3.0, color="k", alpha=.07); ax[1].set_xscale("log"); ax[1].set_xlim(0.1, CFG.fs/2)
ax[1].set_xlabel("Hz"); ax[1].set_title("band responses (dotted = init)"); ax[1].legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.show()
print("wavelet loss after training: %.2e" % wavelet_loss(model.swt.h0).item())
if model.ht is not None:
    with torch.no_grad(): _bp, _bn = [t.cpu().numpy() for t in model.ht.thresholds()]; _g0 = model.ht.gain_at_zero().cpu().numpy()
    print(f"HT ({'nonneg, b = softplus(raw)' if model.ht.nonneg else 'unconstrained DeSpaWN'}) bands {model.band_names}")
    print("   b+          :", np.round(_bp, 3))
    print("   b-          :", np.round(_bn, 3))
    print("   gain at x→0 :", np.round(_g0, 3), "(> 1 = amplifies small coefficients)")
print("encoder θ+ :", np.round(F.softplus(model.enc_th_pos_raw).detach().cpu().numpy(), 3))
print("encoder θ- :", np.round(F.softplus(model.enc_th_neg_raw).detach().cpu().numpy(), 3))