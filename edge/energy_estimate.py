#!/usr/bin/env python3
"""Hardware-free energy ESTIMATES per decision for every headline biometric model, using the counting methods SNN papers
use when no hardware is measured. Saved so they can later be compared with energy MEASURED on the Raspberry Pi
(edge/pi_bench.py, METRICS_REPORT.md metrics 6-10).

Counted per 10 s window (500 samples at 50 Hz), Zenodo across-days setup, fold 0, seed 0:
  params / model size     trainable parameters, bytes at fp32 and int8
  MACs                    multiply-accumulates if every layer is computed densely (how PyTorch/NumPy on a CPU run it)
  SynOps (ACs)            spike-driven accumulates: (number of spikes) x (fan-out); only for the event-driven SNN
  neuron updates          membrane updates per step (leak multiply + add + compare)
  other FLOPs             element-wise non-linear ops (sigmoid, tanh, exp, abs, log), counted separately
Energy models applied to those counts:
  E1  45 nm fp32 compute only (Horowitz, ISSCC 2014, Fig. 1.1.9): MAC = 3.7 + 0.9 = 4.6 pJ, AC = 0.9 pJ, other FLOP = 4.6 pJ
  E2  45 nm int8 compute only (same table): MAC = 0.2 + 0.03 = 0.23 pJ, AC = 0.03 pJ, other = 0.23 pJ
  E3  E1 + weight reads from SRAM: 5 pJ per 32-bit weight read (Horowitz: 10 pJ per 64-bit access, 8 KB cache).
      Recurrent layers read every weight once per time step; CNN weights are read once per window (perfect spatial
      reuse); the event-driven SNN reads one weight per SynOp.
  E4  Loihi (Davies et al., IEEE Micro 2018, 14 nm, 0.75 V): 23.6 pJ per synaptic spike op, 81 pJ per active and
      52 pJ per inactive neuron update, 1.7 pJ per within-tile spike; the wavelet front end and readout are charged at
      E1 because they are not spiking.
The SNN is scored twice: dense (spikes multiplied like numbers, as on a CPU today) and event-driven (only spikes do
work, as on neuromorphic hardware or a spike-skipping CPU kernel). Spike rates are MEASURED on the test windows by
retraining fold 0 / seed 0 with the exact stored configuration; the retrained model's test predictions are compared
with the stored run's to show it is the same model.
Also measured: single-thread CPU latency per window on this machine (not the Pi).

Usage: python -I edge/energy_estimate.py <data_root> <out.json> [threads]
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, next(p for p in (os.path.join(HERE, "..", "biometrics"), HERE) if os.path.exists(os.path.join(p, "bio_run.py"))))
import bio_run as br

PJ = {"E1": {"mac": 4.6, "ac": 0.9, "other": 4.6}, "E2": {"mac": 0.23, "ac": 0.03, "other": 0.23}}
SRAM_PJ_PER_WEIGHT = 5.0
LOIHI = {"synop": 23.6, "neuron_active": 81.0, "neuron_inactive": 52.0, "spike": 1.7}
SOURCES = {"horowitz2014": "M. Horowitz, 'Computing's energy problem (and what we can do about it)', ISSCC 2014, Fig. 1.1.9 (45 nm, 0.9 V): "
                           "int add 8b 0.03 pJ / 32b 0.1 pJ; int mult 8b 0.2 / 32b 3.1; FAdd 16b 0.4 / 32b 0.9; FMult 16b 1.1 / 32b 3.7; "
                           "cache 64-bit 8 KB 10 pJ, 32 KB 20 pJ, 1 MB 100 pJ; DRAM 1.3-2.6 nJ",
           "davies2018": "M. Davies et al., 'Loihi: a neuromorphic manycore processor with on-chip learning', IEEE Micro 2018 (14 nm, 0.75 V): "
                         "synaptic spike op (min) 23.6 pJ; neuron update active/inactive 81/52 pJ; within-tile spike 1.7 pJ"}

def frontend_counts(cfg, T):
    """Learnable SWT (causal a-trous, 2 filters x taps per level per sample) + HT shrink + per-band gain."""
    if not cfg.use_wavelet: return {"mac": 0, "other": 0}
    C = cfg.levels + 1 - len(cfg.drop_bands)
    swt = cfg.levels*2*cfg.taps*T
    ht = C*T*6 if cfg.use_ht else 0            # two sigmoids + products per band sample
    return {"mac": swt + C*T, "other": ht}

def snn_counts(cfg, T, K, enc_rate, hid_rate, amp=True):
    C = cfg.levels + 1 - len(cfg.drop_bands) if cfg.use_wavelet else 1; H = cfg.hidden
    Ts = -(-T // cfg.snn_decim); fe = frontend_counts(cfg, T)
    n_in = 2*C; in_spikes = enc_rate*n_in*Ts; hid_spikes = hid_rate*H*Ts
    dense = {"mac": fe["mac"] + Ts*(n_in*H + H*H + H*K) + Ts*K + Ts*(n_in + H) + (2*K if amp else 0),
             "ac": Ts*(2*n_in + 2*H), "other": fe["other"]}
    event = {"mac": fe["mac"] + Ts*K + Ts*(n_in + H) + (2*K if amp else 0),     # leak multiplies + readout leak stay MACs
             "ac": in_spikes*H + hid_spikes*H + hid_spikes*K + Ts*(2*n_in + 2*H), "other": fe["other"],
             "synops": in_spikes*H + hid_spikes*(H + K)}
    weights_dense = Ts*(n_in*H + H*H + H*K)
    return {"steps": Ts, "input_spikes": in_spikes, "hidden_spikes": hid_spikes, "neuron_updates": Ts*(n_in + H),
            "dense": dense, "event": event, "weight_reads_dense": weights_dense, "weight_reads_event": event["synops"],
            "frontend": fe}

def gru_counts(cfg, T, K, h):
    C = cfg.levels + 1 - len(cfg.drop_bands); Ts = -(-T // cfg.snn_decim); fe = frontend_counts(cfg, T)
    per = 3*(C*h + h*h) + 3*h                  # three gates' matmuls + element-wise gate products
    return {"steps": Ts, "hidden": h, "dense": {"mac": fe["mac"] + Ts*per + h*K + 2*K, "ac": Ts*3*h, "other": fe["other"] + Ts*3*h},
            "weight_reads_dense": Ts*3*(C*h + h*h) + h*K}

def cnn_counts(T, K, width=16, n_amp=2, fs=50):
    import pywt
    H, W = 32, -(-T // 4); mac = 0; w = width; chans = [(1, w), (w, 2*w), (2*w, 4*w)]; h_, w_ = H, W; params = 0
    for i, (ci, co) in enumerate(chans):
        mac += h_*w_*co*ci*9 + h_*w_*co*2; params += co*ci*9 + co + 2*co        # conv + BN(affine at inference)
        if i < 2: h_, w_ = h_//2, w_//2
    mac += (4*w + n_amp)*K; params += (4*w + n_amp)*K + K
    freqs = np.geomspace(0.5, 8, 32); sc = pywt.central_frequency("morl")*fs/freqs
    lens = np.array([16*s + 1 for s in sc])                                   # morl support [-8, 8] sampled at each scale
    cwt_direct_strided = int(np.sum(lens)*W)                                  # only every 4th output sample is kept
    nfft = int(2**np.ceil(np.log2(T + lens.max()))); cwt_fft = int(32*(2*5*nfft*np.log2(nfft)/4 + nfft))  # ~real FFT pair + product
    cwt = min(cwt_direct_strided, cwt_fft)
    return {"params": params, "dense": {"mac": mac + cwt, "ac": 0, "other": 32*W*2},   # abs + log per scalogram pixel
            "cwt_macs": cwt, "cwt_direct_strided_macs": cwt_direct_strided, "cwt_fft_flops_est": cwt_fft,
            "weight_reads_dense": params}

def energy(c, kind):
    e = {}
    for name, p in PJ.items(): e[name] = (c[kind]["mac"]*p["mac"] + c[kind].get("ac", 0)*p["ac"] + c[kind].get("other", 0)*p["other"])/1e6
    reads = c.get("weight_reads_event" if kind == "event" else "weight_reads_dense", 0)
    e["E3"] = e["E1"] + reads*SRAM_PJ_PER_WEIGHT/1e6
    return e                                                                    # microjoules per window

def loihi_energy(c, hid_rate, enc_rate):
    fe = c["frontend"]; host = (fe["mac"]*PJ["E1"]["mac"] + fe["other"]*PJ["E1"]["other"])/1e6
    spikes = c["input_spikes"] + c["hidden_spikes"]; upd = c["neuron_updates"]
    act = spikes; inact = upd - act
    return host + (c["event"]["synops"]*LOIHI["synop"] + act*LOIHI["neuron_active"] + inact*LOIHI["neuron_inactive"] + spikes*LOIHI["spike"])/1e6

def main():
    root, out = sys.argv[1], sys.argv[2]; th = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    import torch; torch.set_num_threads(th)
    g = br.load_notebook_code(); g["VARIANTS"].update(br.BIO_VARIANTS); br.make_amp_fusion(g); br.make_packet(g)
    X, y, S, V, U = br.load_zenodo_bio(root, 10, 2)
    AMP = br.AMP if br.AMP is not None else np.stack([X.mean(1), np.log(X.std(1) + 1e-6)], 1)   # as bio_run.main
    X =((X - X.mean(1, keepdims=True))/(X.std(1, keepdims=True) + 1e-6)).astype(np.float32)
    CFG = g["CFG"]; CFG.n_classes = K = len(U); CFG.win_sec = 10; T = X.shape[1]
    tr, va, te = br.session_folds(y, S)[0]
    res_dir = f"{root}/results/bio_zenodo_bio_w10"
    R = {"setup": {"dataset": "zenodo_bio (across days)", "fold": 0, "seed": 0, "window_samples": int(T), "fs": 50, "classes": K,
                   "test_windows": int(len(te))}, "sources": SOURCES, "pj": {**PJ, "sram_per_weight": SRAM_PJ_PER_WEIGHT, "loihi": LOIHI},
         "models": {}}
    def latency(fn, n=50):
        torch.set_num_threads(1); xb = None; fn(1); t0 = time.perf_counter()
        for _ in range(n): fn(1)
        torch.set_num_threads(th); return 1000*(time.perf_counter() - t0)/n
    for v in ("x_tau_amp", "x_aug_tau_amp", "x_lean_tau_amp", "x_gru_amp"):
        c = g["variant_config"](v, 0, base=CFG); c.n_classes = K
        if "--smoke" in sys.argv: c.epochs, c.patience, c.max_windows_per_epoch = 1, 1, 64
        mu, sd = AMP[tr].mean(0), AMP[tr].std(0) + 1e-6
        Xa = np.concatenate([X, ((AMP - mu)/sd).astype(np.float32)], 1); g["AMP_COLS"] = AMP.shape[1]
        t0 = time.time(); outm = g["train"](c, tr, va, verbose=False, Xa=Xa, ya=y); m = outm["last"]; m.eval()
        Pt = g["predict"](m, te, Xa=Xa)
        stored = json.load(open(f"{res_dir}/{v}_f0_ts0.json")); Ps = np.array(stored["last"]["p_te"])
        same = {"argmax_agreement": float((Pt.argmax(1) == Ps.argmax(1)).mean()), "max_abs_prob_diff": float(np.abs(Pt - Ps).max())}
        er, hr, n = 0.0, 0.0, 0
        with torch.no_grad():
            for b in np.array_split(te, max(1, len(te)//128)):
                _, aux = m(torch.tensor(Xa[b])); er += float(aux["enc_rate"])*len(b); hr += float(aux["hid_rate"])*len(b); n += len(b)
        er, hr = er/n, hr/n
        lat = latency(lambda k: m(torch.tensor(Xa[te[:k]])))
        if c.arch == "snn":
            cnt = snn_counts(c, T, K, er, hr); params = int(g["n_params"](m))
            rec = {"params": params, "bytes_fp32": 4*params, "bytes_int8": params, "measured_enc_rate": er, "measured_hid_rate": hr,
                   "counts": cnt, "energy_uJ_dense": energy(cnt, "dense"), "energy_uJ_event": energy(cnt, "event"),
                   "energy_uJ_loihi": loihi_energy(cnt, hr, er)}
        else:
            cnt = gru_counts(c, T, K, m.base.h if hasattr(m, "base") else m.h); params = int(g["n_params"](m))
            rec = {"params": params, "bytes_fp32": 4*params, "bytes_int8": params, "counts": cnt, "energy_uJ_dense": energy(cnt, "dense")}
        rec.update({"cpu_ms_per_window_1thread_pytorch": lat, "retrain_minutes": (time.time() - t0)/60, "same_as_stored_run": same})
        R["models"][v] = rec; print(v, json.dumps({k: rec[k] for k in rec if k.startswith(("energy", "measured", "params", "same"))}, default=float), flush=True)
    cnt = cnn_counts(T, K); R["models"]["cwt_cnn_amp"] = {"params": cnt["params"], "bytes_fp32": 4*cnt["params"], "bytes_int8": cnt["params"],
                                                         "counts": cnt, "energy_uJ_dense": energy(cnt, "dense")}
    from sklearn.svm import SVC; from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import make_pipeline
    F = np.concatenate([br.beat_features(X, 50), AMP], 1); d = F.shape[1]
    svm = make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=0)).fit(F[tr], y[tr])
    nsv = int(svm[-1].support_vectors_.shape[0]); pairs = K*(K - 1)//2
    sc = {"dense": {"mac": nsv*d + d + pairs*2*(nsv//K + 1) + pairs*2, "ac": 0, "other": nsv + pairs}, "weight_reads_dense": nsv*d,
          "support_vectors": nsv, "features": d, "note": "feature extraction (band-pass, peak finding, beat template) not counted"}
    R["models"]["beat_svm_amp"] = {"params": nsv*d, "counts": sc, "energy_uJ_dense": energy(sc, "dense")}
    for k, r in R["models"].items():
        r["energy_uJ_per_5_window_decision"] = {kk: 5*vv for kk, vv in (r.get("energy_uJ_event") or r["energy_uJ_dense"]).items()}
    json.dump(R, open(out, "w"), indent=1, default=float); print("saved", out)

if __name__ == "__main__": main()
