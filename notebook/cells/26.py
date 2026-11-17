# P1.9 Label-noise check: interval-irregularity features per class (labels from chart notes up to 15 min from the waveform)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.2))
for j, name in enumerate(FEAT_NAMES):
    for c, cn in enumerate(CLASS_NAMES):
        v = FEATS[y == c, j]; v = v[np.isfinite(v)]
        ax[j].hist(np.clip(v, 0, np.nanpercentile(FEATS[:, j], 99)), bins=60, alpha=.5, density=True, label=f"{cn} (n={len(v)})")
    ax[j].set_title(name); ax[j].legend(fontsize=7)
ax[0].axvline(REG_RULE["cv_max"], color="g", ls=":"); ax[0].axvline(IRREG_RULE["cv_min"], color="r", ls=":")
ax[2].axvline(REG_RULE["frac_max"], color="g", ls=":"); ax[2].axvline(IRREG_RULE["frac_min"], color="r", ls=":")
plt.tight_layout(); plt.show()
with np.errstate(invalid="ignore"):
    _reg, _irr = looks_regular(FEATS), looks_irregular(FEATS)
_ok = np.isfinite(FEATS[:, 0])
print(f"rule: regular = {REG_RULE}, irregular = {IRREG_RULE} (heuristic thresholds); windows without enough peaks: {(~_ok).mean():.1%}")
for c, cn in enumerate(CLASS_NAMES):
    m = (y == c) & _ok
    print(f"  {cn:6s}: {m.sum():6d} windows with features | look regular {_reg[m].mean():6.1%} | look irregular {_irr[m].mean():6.1%}")
if AF_IDX is not None:
    _dis = label_disagrees(np.arange(len(y)))
    print(f"AF-labelled windows that look regular: {_reg[(y == AF_IDX) & _ok].mean():.1%} | "
          f"{CLASS_NAMES[SR_IDX]}-labelled windows that look irregular: {_irr[(y == SR_IDX) & _ok].mean():.1%} | "
          f"strong disagreements: {_dis.sum()} windows ({_dis.mean():.1%}) in {len(set(groups[_dis]))} subjects")
    print("LABEL_CLEAN_TRAIN =", LABEL_CLEAN_TRAIN, "-> when True, variant 'full_clean' drops these windows from TRAIN and VAL "
          "only. Test folds stay uncleaned, so the classifier is never scored on data filtered by its own rule. Note: the "
          "'irreg_lr' baseline uses the same features.")
