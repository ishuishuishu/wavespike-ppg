# §9a Variants to run + time estimate (printed BEFORE the ablation is launched)
RUN_VARIANTS = ["full", "irreg_lr", "gru_twin", "snn_raw", "fixed_wavelet", "ht_despawn", "lean", "aug", "spkdrop", "h32"]
# also available: "shared_filter", "no_ht", "reg_all"
if LABEL_CLEAN_TRAIN: RUN_VARIANTS.append("full_clean")
REPEATS = [(cs, ts) for cs in CV_SEEDS for ts in TRAIN_SEEDS]
# v3.1: optional overrides for screening runs (unset = exactly the plan above). WAVESPIKE_VARIANTS="a,b"  WAVESPIKE_REPEATS="0:0,3:1" (cv:ts)
if os.environ.get("WAVESPIKE_VARIANTS"): RUN_VARIANTS = os.environ["WAVESPIKE_VARIANTS"].split(",")
if os.environ.get("WAVESPIKE_REPEATS"): REPEATS = [tuple(int(a) for a in r.split(":")) for r in os.environ["WAVESPIKE_REPEATS"].split(",")]
for _cs, _ts in REPEATS:
    if _cs not in FOLDS_BY_CV: FOLDS_BY_CV[_cs] = make_folds(STRAT, groups, CFG.n_folds, seed=_cs)

def time_train_batches(cfg, n=3):
    set_seed(0); mdl = build_model(cfg).to(DEVICE); mdl.train()
    opt = torch.optim.AdamW(mdl.parameters(), lr=1e-4); b = FOLDS[0][0][:cfg.batch]
    xb, yb = torch.tensor(X[b], device=DEVICE), torch.tensor(y[b], device=DEVICE)
    def step():
        lg, _ = mdl(xb); loss = F.cross_entropy(lg, yb); opt.zero_grad(); loss.backward(); opt.step()
    step()
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n): step()
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    return (time.time() - t0)/n

ep_sec_full = hist[0]["sec"]                       # first epoch of the §6 run (train + validation)
epochs_est = len(hist)                             # epochs §6 needed (early stopping)
n_ext = len(XE) if XE is not None else 0
_t0 = time.time(); predict(model, FOLDS[0][2][:256]); _pw = (time.time() - _t0)/min(256, len(FOLDS[0][2]))
eval_sec_full = 2*(len(FOLDS[0][2]) + n_ext)*_pw    # best + last checkpoint on test + external
tb = {}
for v in RUN_VARIANTS:
    if VARIANTS[v].get("_kind") == "lr": continue
    tb[v] = time_train_batches(variant_config(v, 0))
if "full" not in tb: tb["full"] = time_train_batches(variant_config("full", 0))      # reference for the relative timings
est = {}
for v in RUN_VARIANTS:
    est[v] = 0.05 if VARIANTS[v].get("_kind") == "lr" else (epochs_est*ep_sec_full + eval_sec_full) * tb[v]/tb["full"] / 60
todo = {v: sum(not os.path.exists(run_path(v, cs, ts, k)) for cs, ts in REPEATS for k in range(CFG.n_folds)) for v in RUN_VARIANTS}
total_h = sum(est[v]*todo[v] for v in RUN_VARIANTS)/60
per_repeat_h = sum(est[v] for v in RUN_VARIANTS)*CFG.n_folds/60
left_h = MAX_SESSION_H - (time.time() - NOTEBOOK_T0)/3600
print(f"TIME ESTIMATE from the first §6 epoch ({ep_sec_full:.0f} s/epoch for 'full') × {epochs_est} epochs/run (as in §6):")
for v in RUN_VARIANTS:
    print(f"  {v:14s} ≈ {est[v]:6.1f} min/run" + (f"  (batch time × {tb[v]/tb['full']:.2f} vs full)" if v in tb else "") +
          f" | runs to do {todo[v]:3d}")
print(f"  total remaining ≈ {total_h:.1f} h | one repeat (all variants × {CFG.n_folds} folds) ≈ {per_repeat_h:.1f} h | "
      f"{len(REPEATS)} repeats planned")
print(f"  this session has ≈ {left_h:.1f} h left before MAX_SESSION_H -> ≈ {max(0, left_h)/max(per_repeat_h, 1e-9):.1f} repeats; "
      f"≈ {max(1, math.ceil(total_h/MAX_SESSION_H))} sessions in total")
if total_h > left_h:
    print("  ⚠ The full ablation doesn't fit in one session. It is resumable: re-run the notebook in new sessions. To cut "
          "the time instead, shorten RUN_VARIANTS or TRAIN_SEEDS, but no improvement claim is made with fewer than 6 repeats.")
