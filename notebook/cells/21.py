# §8 Spike raster for one test window + per-band encoder rates over 128 test windows (dead-band check, P1.7)
with torch.no_grad(): _, aux = model(torch.tensor(X[te[:1]], device=DEVICE), record=True)
enc, hid = aux["S_in"].numpy(), aux["SH"].numpy(); tt = np.arange(enc.shape[0])*model.decim/CFG.fs
fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
ax[0].plot(np.arange(T_WIN)/CFG.fs, X[te[0]], lw=.8); ax[0].set_title(f"true: {CLASS_NAMES[y[te[0]]]}")
lbl = [f"+{n}" for n in model.band_names] + [f"−{n}" for n in model.band_names]
for k in range(enc.shape[1]): ax[1].plot(tt[enc[:, k] > 0], np.full((enc[:, k] > 0).sum(), k), "|", ms=4)
ax[1].set_yticks(range(len(lbl))); ax[1].set_yticklabels(lbl, fontsize=6)
for k in range(hid.shape[1]): ax[2].plot(tt[hid[:, k] > 0], np.full((hid[:, k] > 0).sum(), k), "|", ms=2, color="k")
ax[2].axvline(model.w0*model.decim/CFG.fs, color="r", ls="--", lw=1, label="readout starts (filter warm-up)"); ax[2].legend(fontsize=7)
ax[2].set_xlabel("s"); plt.tight_layout(); plt.show()
with torch.no_grad(): _, aux = model(torch.tensor(X[te[:128]], device=DEVICE))
rates = aux["enc_rate_ch"].cpu().numpy()
print("per-band encoder rate (128 test windows):", {k: round(float(v), 4) for k, v in zip(lbl, rates)})
print("dead bands (rate of both + and − < 0.005):",
      [n for i, n in enumerate(model.band_names) if max(rates[i], rates[i + model.C]) < 0.005] or "none")
