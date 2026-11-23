#!/usr/bin/env python3
"""Step 11 Part C (STEP11_PREDECLARED_PROTOCOL.md): export one trained run to NumPy, check parity on its full test set,
and build the Raspberry Pi bundle.

Usage: python -I step11_export_parity.py <run json> <out dir>
  <run json>: a step11_run.py result (its *_last.pt checkpoint must sit next to it).
Writes <out dir>/: snn_bio_model.npz, step11_rpi_infer.py, samples.npz, parity_summary.json, README.md, run_pi_selfcheck.sh
"""
import json, os, sys, hashlib, shutil, time, platform
from dataclasses import asdict
import numpy as np
REPO = os.environ.get("WAVESPIKE_REPO", os.path.join(os.path.expanduser("~"), "capstone_metrics_run", "wavespike-ppg"))
sys.path.insert(0, os.path.join(REPO, "biometrics")); sys.path.insert(0, os.path.join(REPO, "edge"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bio_run as B
import step11_rpi_infer as R
import rpi_infer as TEAM                       # the teammate's NumPy kernel, for a line-for-line cross-check

def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

def main():
    run_json, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    rec = json.load(open(run_json)); ck = run_json[:-5] + "_last.pt"
    vname = rec["variant"]; base_v = vname[:-5] if vname.endswith("_more") else vname; k, ts = rec["fold"], rec["seed"]
    root = os.environ["WAVESPIKE_DATA_ROOT"]
    import torch; torch.set_num_threads(4)
    state = None
    if not os.path.exists(ck):     # a bio_run.py --save-model run (Part A): weights and amp normalisation in one .pt file
        ck = f"{root}/models/step11/{os.path.basename(os.path.dirname(os.path.abspath(run_json)))}__{vname}_f{k}_ts{ts}.pt"
        sv = torch.load(ck, map_location="cpu", weights_only=False)
        assert (sv["variant"], sv["fold"], sv["seed"]) == (vname, k, ts), "saved model does not belong to this run"
        state = sv["last_state"]; rec["amp_mu"], rec["amp_sd"] = sv["amp_norm"]["mean"], sv["amp_norm"]["std"]
    g = B.load_notebook_code(); g["VARIANTS"].update(B.BIO_VARIANTS); B.make_amp_fusion(g); B.make_packet(g)
    more = vname.endswith("_more")
    d = np.load(f"{root}/processed/" + ("step11_zenodo_more_50hz_10s.npz" if more else "zenodo_bio_50hz_10s.npz"), allow_pickle=True)
    Xraw, y, S, V, U = d["X"].astype(np.float32), d["y"], d["sess"], d["vid"], [str(u) for u in d["users"]]
    E = ~d["is_extra"] if more else np.ones(len(y), bool)
    AMP = np.stack([Xraw.mean(1), np.log(Xraw.std(1) + 1e-6)], 1)
    X = ((Xraw - Xraw.mean(1, keepdims=True))/(Xraw.std(1, keepdims=True) + 1e-6)).astype(np.float32)
    CFG = g["CFG"]; CFG.n_classes = len(U); CFG.win_sec = 10
    tr, va, te = B.session_folds(y, S, E=E if more else None)[k]
    assert y[te].tolist() == rec["y_te"] and V[te].tolist() == rec["vid_te"], "test partition differs from the run"
    assert np.all(np.diff(te) > 0), "test indices must be increasing (used for the sample lookup)"
    c = g["variant_config"](base_v, ts, base=CFG); c.n_classes = len(U)
    mu, sd = np.asarray(rec["amp_mu"], np.float32), np.asarray(rec["amp_sd"], np.float32)
    Xa = np.concatenate([X, ((AMP - mu)/sd).astype(np.float32)], 1); g["AMP_COLS"] = 2
    model = g["build_model"](c); model.load_state_dict(state if state is not None else torch.load(ck, map_location="cpu"), strict=True); model.eval()
    lg_t = g["predict_logits"](model, Xa, te); P_t = g["softmax_np"](lg_t)
    lg_v = g["predict_logits"](model, Xa, va); P_v = g["softmax_np"](lg_v)
    d_run = float(np.abs(P_t - np.asarray(rec["last"]["p_te"])).max())
    assert d_run <= 2e-6, f"checkpoint does not reproduce the run's saved test probabilities (max |dp| {d_run})"
    met = B.bio_metrics(P_v, y[va], V[va], P_t, y[te], V[te], 5); tau = met["tau"]
    # ---- export
    sdict = {kk: v.detach().cpu().numpy() for kk, v in model.state_dict().items()}
    gb = lambda kk: sdict["base." + kk]
    exp = {"h0": sdict["swt.h0"], "enc_gain": gb("enc_gain"), "enc_beta_raw": gb("enc_beta_raw"),
           "enc_th_pos_raw": gb("enc_th_pos_raw"), "enc_th_neg_raw": gb("enc_th_neg_raw"),
           "fc_in.weight": gb("fc_in.weight"), "fc_in.bias": gb("fc_in.bias"), "fc_rec.weight": gb("fc_rec.weight"),
           "hid_beta_raw": gb("hid_beta_raw"), "fc_out.weight": gb("fc_out.weight"), "fc_out.bias": gb("fc_out.bias"),
           "out_beta_raw": gb("out_beta_raw"), "amp.weight": sdict["amp.weight"], "amp.bias": sdict["amp.bias"],
           "amp_mu": mu, "amp_sd": sd, "tau": np.float64(tau), "n_agg": np.int64(5),
           "config": json.dumps({**asdict(c), "class_names": U, "dataset": "zenodo_23597128_v1_bio",
                                 "variant": vname, "fold": k, "train_seed": ts, "source_ckpt_sha256": sha(ck)})}
    assert c.ht_nonneg and model.base.ht is not None
    exp["ht_b_pos_raw"], exp["ht_b_neg_raw"] = gb("ht.b_pos_raw"), gb("ht.b_neg_raw")
    mpath = f"{out}/snn_bio_model.npz"; np.savez(mpath, **exp)
    net = R.WaveSpikeBioNP(mpath)
    # ---- cross-check against the teammate's kernel (base logits, no amplitude head) on 50 test windows
    tpath = f"{out}/_team_export_check.npz"
    np.savez(tpath, **{kk: v for kk, v in exp.items() if kk not in ("amp.weight", "amp.bias", "amp_mu", "amp_sd", "tau", "n_agg")})
    team = TEAM.WaveSpikeNP(tpath); dteam = 0.0
    for i in te[:50]:
        ours = net.window_logits(Xraw[i]) - (net.W_amp @ net.amp_features(Xraw[i]) + net.b_amp)
        dteam = max(dteam, float(np.abs(ours - team.classify_window(Xraw[i])[3]).max()))
    os.remove(tpath)
    # ---- full test-set parity
    t0 = time.perf_counter(); lg_n, margins = [], []
    for i in te:
        st = {}; lg_n.append(net.window_logits(Xraw[i], st)); margins.append(st["margin"])
    sec = time.perf_counter() - t0
    lg_n = np.stack(lg_n); margins = np.asarray(margins); P_n = g["softmax_np"](lg_n.astype(np.float32))
    dl = np.abs(lg_n - lg_t).max(1); mism = lg_n.argmax(1) != lg_t.argmax(1); near = margins < 1e-5
    Pd_t, yd = B.aggregate(P_t, y[te], V[te], 5); Pd_n, _ = B.aggregate(P_n, y[te], V[te], 5)
    gen_t, imp_t, _ = B.trials(Pd_t, yd); gen_n, imp_n, _ = B.trials(Pd_n, yd)
    summ = {"run": os.path.basename(run_json), "variant": vname, "fold": k, "seed": ts, "n_test_windows": int(len(te)),
            "ckpt_reproduces_run_max_dp": d_run, "teammate_kernel_max_dlogit_50win": dteam,
            "window_argmax_mismatches": int(mism.sum()), "max_abs_dlogit": float(dl.max()),
            "windows_dlogit_gt_1e-4": int((dl > 1e-4).sum()), "windows_threshold_adjacent_lt_1e-5": int(near.sum()),
            "windows_dlogit_gt_1e-4_not_threshold_adjacent": int(((dl > 1e-4) & ~near).sum()),
            "decisions_5win": int(len(yd)), "decision_argmax_mismatches": int((Pd_t.argmax(1) != Pd_n.argmax(1)).sum()),
            "tau_5win_val_far2pct": tau, "test_far_count_torch": int((imp_t >= tau).sum()), "test_far_count_numpy": int((imp_n >= tau).sum()),
            "test_frr_count_torch": int((gen_t < tau).sum()), "test_frr_count_numpy": int((gen_n < tau).sum()),
            "n_impostor": int(len(imp_t)), "n_genuine": int(len(gen_t)), "metrics_5win_torch": {kk: met[kk] for kk in ("eer", "top1", "macro_f1", "far", "frr")},
            "numpy_ms_per_window_nas": 1e3*sec/len(te), "platform": platform.platform(), "numpy": np.__version__,
            "model_file_bytes": os.path.getsize(mpath), "model_sha256": sha(mpath)}
    # ---- supplementary implementation check (erratum E3), reported separately; the PARITY rule below is unchanged.
    # Same weights in float64 on both sides: torch model.double() vs a float64 copy of step11_rpi_infer (parameters
    # upcast before sigmoid/softplus). Equal to ~1e-12 means the NumPy code implements the model exactly and any
    # float32 difference is rounding. torch32 vs torch64 shows how far rounding alone moves the reference model.
    import copy, types
    src64 = open(R.__file__).read().replace("np.float32", "np.float64").replace(
        "d = np.load(path, allow_pickle=False)",
        "d = {kk: (vv.astype(np.float64) if vv.dtype.kind == 'f' else vv) for kk, vv in np.load(path, allow_pickle=False).items()}"
    ).replace("d.files", "d")
    assert "vv.astype(np.float64)" in src64 and "np.float32" not in src64
    R64 = types.ModuleType("step11_rpi_infer_f64"); exec(compile(src64, "step11_rpi_infer_f64", "exec"), R64.__dict__)
    net64 = R64.WaveSpikeBioNP(mpath)
    Xr64 = Xraw.astype(np.float64); X64 = (Xr64 - Xr64.mean(1, keepdims=True))/(Xr64.std(1, keepdims=True) + 1e-6)
    A64 = np.stack([Xr64.mean(1), np.log(Xr64.std(1) + 1e-6)], 1)
    Xa64 = np.concatenate([X64, (A64 - mu.astype(np.float64))/sd.astype(np.float64)], 1)
    m64 = copy.deepcopy(model).double().eval()
    with torch.no_grad():
        lt64 = np.concatenate([m64(torch.tensor(Xa64[b]))[0].numpy() for b in np.array_split(te, max(1, len(te)//128))])
    with np.errstate(over="ignore"):
        ln64 = np.stack([net64.window_logits(Xr64[i]) for i in te])
    d64 = np.abs(ln64 - lt64).max(1); d3264 = np.abs(lg_t.astype(np.float64) - lt64).max(1)
    summ.update({"f64_torch_vs_numpy_max_abs_dlogit": float(d64.max()),
                 "f64_torch_vs_numpy_argmax_mismatches": int((ln64.argmax(1) != lt64.argmax(1)).sum()),
                 "torch_f32_vs_f64_windows_dlogit_gt_1e-4": int((d3264 > 1e-4).sum()),
                 "torch_f32_vs_f64_argmax_mismatches": int((lg_t.argmax(1) != lt64.argmax(1)).sum()),
                 "torch_f32_vs_numpy_f32_windows_dlogit_gt_1e-4": int((dl > 1e-4).sum())})
    ok64 = summ["f64_torch_vs_numpy_max_abs_dlogit"] <= 1e-9 and summ["f64_torch_vs_numpy_argmax_mismatches"] == 0
    # ---- bundle: 40 reference 5-window decisions (first 40 in bio_run.aggregate order) with raw windows
    idx = []
    for v in np.unique(V[te]):
        iv = te[V[te] == v]
        for a in range(0, len(iv) - 5 + 1, 5): idx.append(iv[a:a+5])
    idx = np.array(idx[:40]); W = Xraw[idx.ravel()]
    ref = np.stack([np.exp(np.log(np.clip(P_t[np.searchsorted(te, grp)], 1e-12, 1)).mean(0)) for grp in idx])
    ref = ref/ref.sum(1, keepdims=True)
    np.savez_compressed(f"{out}/samples.npz", windows=W, ref_decision_probs=ref, true_index=y[idx[:, 0]])
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "step11_rpi_infer.py"), f"{out}/step11_rpi_infer.py")
    json.dump(summ, open(f"{out}/parity_summary.json", "w"), indent=1)
    with open(f"{out}/run_pi_selfcheck.sh", "w") as fh:
        fh.write("#!/usr/bin/env bash\n# Runs on the Raspberry Pi with python3 + numpy only. Exit code 0 = all 40 reference decisions reproduced.\n"
                 "set -euo pipefail\ncd \"$(dirname \"$0\")\"\nsha256sum snn_bio_model.npz\n"
                 "python3 -c 'import sys, numpy, platform; print(sys.version.split()[0], numpy.__version__, platform.machine())'\n"
                 "python3 -I step11_rpi_infer.py snn_bio_model.npz samples.npz\n")
    os.chmod(f"{out}/run_pi_selfcheck.sh", 0o755)
    m5 = summ["metrics_5win_torch"]
    with open(f"{out}/README.md", "w") as fh:
        fh.write(f"""# WaveSpike-PPG biometric model, Raspberry Pi bundle (Step 11 Part C)

NumPy-only inference for `{vname}`, fold {k}, training seed {ts} (Zenodo 23597128 v1, {len(U)} people, 10 s windows at 50 Hz).

| File | What |
|---|---|
| `snn_bio_model.npz` | Weights, config, amplitude-feature normalisation, threshold tau, n_agg = 5 (SHA-256 `{summ['model_sha256']}`) |
| `step11_rpi_infer.py` | Inference (needs only numpy): `WaveSpikeBioNP(path).decide(windows)` and `.verify(windows, claimed_index)` |
| `samples.npz` | 40 reference decisions (5 raw test windows each) with the PyTorch decision probabilities |
| `parity_summary.json` | PyTorch vs NumPy parity on the full test set ({summ['n_test_windows']} windows) |
| `run_pi_selfcheck.sh` | Run this on the Pi: reproduces the 40 reference decisions; prints ms per decision |

Input: raw PPG windows (green channel, resampled to 50 Hz), 500 samples each; z-scoring and amplitude features are done inside.
Decision: mean log-probability over 5 consecutive windows; verification accepts if p(claimed) >= tau = {tau:.6g}
(tau set on validation at 2 % FAR).

Test-set result of this run (PyTorch, 5 windows): EER {100*m5['eer']:.2f} %, top-1 {100*m5['top1']:.1f} %,
FAR {100*m5['far']:.2f} %, FRR {100*m5['frr']:.2f} %.
Parity: {summ['window_argmax_mismatches']} window and {summ['decision_argmax_mismatches']} decision argmax mismatches;
max |d logit| {summ['max_abs_dlogit']:.2e}; FAR/FRR counts torch {summ['test_far_count_torch']}/{summ['test_frr_count_torch']},
numpy {summ['test_far_count_numpy']}/{summ['test_frr_count_numpy']}.
Float64 check (both sides in float64, same weights): max |d logit| {summ['f64_torch_vs_numpy_max_abs_dlogit']:.2e},
{summ['f64_torch_vs_numpy_argmax_mismatches']} argmax mismatches. For scale, PyTorch float32 vs PyTorch float64 differ by more than 1e-4
on {summ['torch_f32_vs_f64_windows_dlogit_gt_1e-4']} windows ({summ['torch_f32_vs_f64_argmax_mismatches']} argmax changes): float32 rounding alone moves this spiking model.

No Raspberry Pi timing or energy is claimed by this bundle; `run_pi_selfcheck.sh` measures it on the device.
""")
    print(json.dumps(summ, indent=1))
    ok = summ["window_argmax_mismatches"] == 0 and summ["decision_argmax_mismatches"] == 0 and \
         summ["windows_dlogit_gt_1e-4_not_threshold_adjacent"] == 0 and dteam <= 1e-5 and \
         summ["test_far_count_torch"] == summ["test_far_count_numpy"] and summ["test_frr_count_torch"] == summ["test_frr_count_numpy"]
    print("PARITY", "PASS" if ok else "FAIL", "| IMPLEMENTATION_F64", "PASS" if ok64 else "FAIL")

if __name__ == "__main__": main()
