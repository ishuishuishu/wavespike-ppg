# §6 Main run: fold 0, trained TWICE with identical settings (P0.1 determinism check)
CS0, TS0, K0 = CV_SEEDS[0], TRAIN_SEEDS[0], 0
tr, va, te = FOLDS_BY_CV[CS0][K0]
c0 = variant_config("full", TS0)
FORCE_DETERMINISM_CHECK = False      # True: train fold 0 twice again even if an earlier session already verified it
DET_MARK = f"{RES_DIR}/determinism_check.json"
_ck = f"{CKPT_DIR}/full_cv{CS0}_ts{TS0}_fold{K0}"
_resume = (os.path.exists(DET_MARK) and os.path.exists(run_path("full", CS0, TS0, K0)) and os.path.exists(_ck + "_best.pt")
           and not FORCE_DETERMINISM_CHECK)
RUNS = []
if _resume:                          # earlier session: verified, load the saved fold-0 run instead of training twice more
    _rec = json.load(open(run_path("full", CS0, TS0, K0))); out = {"hist": _rec["hist"], "best_epoch": _rec["best_epoch"]}
    for name in ("best", "last"):
        out[name + "_state"] = torch.load(f"{_ck}_{name}.pt", map_location=DEVICE)
        out[name] = build_model(c0).to(DEVICE); out[name].load_state_dict(out[name + "_state"]); out[name].eval()
    RUNS.append({"out": out, "P_best": predict(out["best"], te), "P_last": predict(out["last"], te), "sec": _rec["minutes"]*60})
    print("P0.1 determinism was verified in an earlier session:", json.load(open(DET_MARK)), "| loaded the saved fold-0 run")
for rep in range(0 if _resume else 2):
    print(f"--- training run {rep+1}/2 (full, fold seed {CS0}, train seed {TS0}, fold {K0}) ---")
    try:
        t0 = time.time(); out = train(c0, tr, va, verbose=(rep == 0))
    except RuntimeError as e:
        if "deterministic" in str(e).lower():
            print("P0.1 NONDETERMINISTIC OP (torch refused to run it in deterministic mode):\n   ", str(e).splitlines()[0])
        raise
    RUNS.append({"out": out, "P_best": predict(out["best"], te), "P_last": predict(out["last"], te), "sec": time.time() - t0})

if not _resume:
    same_pred = all(np.array_equal(RUNS[0][k], RUNS[1][k]) for k in ("P_best", "P_last"))
    same_hist = [h["val_loss"] for h in RUNS[0]["out"]["hist"]] == [h["val_loss"] for h in RUNS[1]["out"]["hist"]]
    same_w = all(torch.equal(RUNS[0]["out"]["last_state"][k], RUNS[1]["out"]["last_state"][k]) for k in RUNS[0]["out"]["last_state"])
    m_a, m_b = metrics(y[te], RUNS[0]["P_best"]), metrics(y[te], RUNS[1]["P_best"])
    print(f"P0.1 determinism: test AUROC {m_a['auroc']:.6f} vs {m_b['auroc']:.6f} | identical predictions: {same_pred} | "
          f"identical val-loss curves: {same_hist} | identical weights: {same_w}")
    if not (same_pred and same_hist and same_w):
        d = next((i for i, (a, b) in enumerate(zip(RUNS[0]["out"]["hist"], RUNS[1]["out"]["hist"])) if a["val_loss"] != b["val_loss"]), None)
        raise AssertionError(f"training is not deterministic (first differing epoch: {d}). No op raised in deterministic mode, so "
                             "the difference comes from outside torch's checks (e.g. CUBLAS_WORKSPACE_CONFIG set after CUDA "
                             "started: restart the session).")
    json.dump({"test_auroc_run1": m_a["auroc"], "test_auroc_run2": m_b["auroc"], "identical": True, "torch": torch.__version__,
               "device": str(DEVICE), "date": time.strftime("%Y-%m-%d %H:%M")}, open(DET_MARK, "w"))

out = RUNS[0]["out"]; model, hist = out["best"], out["hist"]
print(f"best epoch {out['best_epoch']} (lowest val loss) of {len(hist)} run")
report(metrics(y[te], RUNS[0]["P_best"]), "TEST fold 0 [best-val-loss checkpoint]")
report(metrics(y[te], RUNS[0]["P_last"]), "TEST fold 0 [last-epoch checkpoint]")
fig, ax = plt.subplots(1, 2, figsize=(12, 3))
ax[0].plot([h["loss"] for h in hist], label="train loss"); ax[0].plot([h["val_loss"] for h in hist], label="val loss")
ax[0].axvline(out["best_epoch"], color="k", ls=":"); ax[0].legend(); ax[0].set_xlabel("epoch")
ax[1].plot([h["val_auroc"] for h in hist], label="val AUROC"); ax[1].plot([h["val_f1"] for h in hist], label="val F1")
ax[1].axvline(out["best_epoch"], color="k", ls=":"); ax[1].legend(); ax[1].set_xlabel("epoch"); plt.tight_layout(); plt.show()

# This run is (full, CS0, TS0, fold 0) of the ablation: save it, so §9 doesn't train it again.
if not os.path.exists(run_path("full", CS0, TS0, K0)):
    tag = f"{CKPT_DIR}/full_cv{CS0}_ts{TS0}_fold{K0}"
    torch.save(out["best_state"], tag + "_best.pt"); torch.save(out["last_state"], tag + "_last.pt")
    ents = {}
    for name in ("best", "last"):
        PE = predict(out[name], np.arange(len(XE)), Xa=XE) if XE is not None else None
        ents[name] = _probs_entry(RUNS[0]["P_" + name], PE, te)
    save_run("full", CS0, TS0, K0, te, ents, {"epochs_run": len(hist), "best_epoch": out["best_epoch"], "minutes": RUNS[0]["sec"]/60,
             "hist": [{k_: (round(v_, 5) if isinstance(v_, float) else v_) for k_, v_ in h.items()} for h in hist]})
    if XE is not None:
        print(f"external MIMIC PERform AF [{EXTERNAL_OVERLAP_STATUS}]:", {k: round(v, 3) for k, v in ents["best"]["ext"].items()})
