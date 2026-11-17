#!/usr/bin/env python3
"""Edge benchmark for the exported WaveSpike-PPG model (NumPy + stdlib only; runs on a Raspberry Pi).

One invocation = one process = one repetition (peak RSS is per process). Run it >= 5 times, alternating models/runtimes.
  python3 pi_bench.py --model wavespike_rpi.npz --windows bench_windows.npy --n 1000 --out run1.json
  Energy phases (external meter logs V and I; GPIO pin marks the active window):
  python3 pi_bench.py ... --idle-seconds 60 --active-seconds 60 --gpio-pin 17 --out energy1.json
  python3 pi_bench.py --energy-from run.json --power-csv meter.csv      # E_gross / E_incremental from a meter log (t_s,V,I)
Pin to one core and one thread:  OMP_NUM_THREADS=1 taskset -c 0 python3 pi_bench.py ...
Everything printed is a measurement of THIS process on THIS board, except the op counts, which are an arithmetic proxy.
"""
import argparse, json, os, platform, resource, subprocess, sys, time
import numpy as np

def rss_bytes():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r * 1024 if platform.system() == "Linux" else r

def vcgen(cmd):
    try: return subprocess.run(["vcgencmd", *cmd.split()], capture_output=True, text=True, timeout=2).stdout.strip()
    except Exception: return None

def board_state():
    return {"temp": vcgen("measure_temp"), "arm_clock": vcgen("measure_clock arm"), "throttled": vcgen("get_throttled"),
            "governor": (open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
                         if os.path.exists("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") else None)}

def pct(ms):
    return {"mean": float(np.mean(ms)), "p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95)),
            "p99": float(np.percentile(ms, 99)), "max": float(np.max(ms)), "n": int(len(ms))}

def op_table(net, st, n_dec):
    """Per-decision operation counts. Spike-driven synaptic adds are the SynOps proxy; the rest is listed explicitly."""
    H, K, C2 = net.W_rec.shape[0], net.W_out.shape[0], net.W_in.shape[1]
    T = st["steps"] / n_dec; i_sp, h_sp = st["in_spikes"] / n_dec, st["hid_spikes"] / n_dec
    syn_in, syn_rec, syn_out = i_sp * H, h_sp * H, h_sp * K
    return {"steps_per_decision": T, "input_spikes": i_sp, "hidden_spikes": h_sp,
            "input_sparsity": 1 - i_sp / (T * C2), "hidden_rate": h_sp / (T * H),
            "synops_in_to_hidden": syn_in, "synops_recurrent": syn_rec, "synops_hidden_to_out": syn_out,
            "synops_total_proxy": syn_in + syn_rec + syn_out,
            "bias_adds": T * (H + K), "leak_muls": T * (H + K + C2), "threshold_cmp_reset": T * (H + K + C2),
            "readout_macs": T * K,
            "dense_equivalent_macs_per_step": C2 * H + H * H + H * K,
            "dense_equivalent_macs_total": T * (C2 * H + H * H + H * K)}

def energy_from_log(run_json, csv_path):
    r = json.load(open(run_json)); d = np.loadtxt(csv_path, delimiter=",", skiprows=1)    # columns: t_s, V, I (same clock as the run)
    t, p = d[:, 0], d[:, 1] * d[:, 2]; ph = r["phases"]; out = {}
    def integ(t0, t1):
        m = (t >= t0) & (t <= t1); return float(np.trapz(p[m], t[m])), float(t1 - t0), float(p[m].mean())
    e_idle, T_idle, P_idle = integ(ph["idle_start"], ph["idle_end"]); e_act, T_act, P_act = integ(ph["active_start"], ph["active_end"])
    n = r["active_decisions"]
    out.update(P_idle_W=P_idle, P_active_W=P_act, E_gross_mJ_per_decision=1e3 * e_act / n,
               E_incremental_mJ_per_decision=1e3 * (e_act - P_idle * T_act) / n, decisions=n, active_seconds=T_act)
    print(json.dumps(out, indent=1)); return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wavespike_rpi.npz"); ap.add_argument("--windows", default="bench_windows.npy")
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--idle-seconds", type=float, default=0); ap.add_argument("--active-seconds", type=float, default=0)
    ap.add_argument("--gpio-pin", type=int, default=None); ap.add_argument("--out", default="bench_run.json")
    ap.add_argument("--energy-from"); ap.add_argument("--power-csv")
    a = ap.parse_args()
    if a.energy_from: return energy_from_log(a.energy_from, a.power_csv)
    rss0 = rss_bytes()
    import rpi_infer                                   # same directory; never imports torch/scipy/sklearn
    rss_imports = rss_bytes()
    net = rpi_infer.WaveSpikeNP(a.model); W = np.load(a.windows).astype(np.float32); rss_load = rss_bytes()
    npz = np.load(a.model); arrs = {k: npz[k] for k in npz.files if k != "config"}
    params = int(sum(v.size for v in arrs.values())); state_bytes = int(sum(v.nbytes for v in arrs.values()))
    res = {"model": a.model, "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
           "machine": platform.machine(), "threads_env": os.environ.get("OMP_NUM_THREADS"), "board_before": board_state(),
           "storage": {"artifact_bytes": os.path.getsize(a.model), "params": params, "state_bytes_as_stored": state_bytes},
           "rss_bytes": {"start": rss0, "after_imports": rss_imports, "after_load": rss_load}}
    gpio = None
    if a.gpio_pin is not None:
        try:
            import RPi.GPIO as G; G.setmode(G.BCM); G.setup(a.gpio_pin, G.OUT); G.output(a.gpio_pin, 0); gpio = G
        except Exception as e: res["gpio_error"] = str(e)
    for k in range(min(a.warmup, 10 * len(W))): net.classify_window(W[k % len(W)])
    # latency: whole decision (z-score -> SWT -> HT -> gain -> encoder -> SNN -> readout), and the front end alone
    tl, tf = np.empty(a.n), np.empty(a.n)
    for k in range(a.n):
        x = W[k % len(W)]
        t0 = time.perf_counter_ns(); net.encoder_input(x); t1 = time.perf_counter_ns(); net.classify_window(x); t2 = time.perf_counter_ns()
        tf[k] = (t1 - t0) / 1e6; tl[k] = (t2 - t1) / 1e6
    res["latency_ms"] = {"decision_L1_full_pipeline": pct(tl), "front_end_only": pct(tf), "snn_only_approx": pct(np.maximum(tl - tf, 0))}
    st = {}; nd = min(a.n, 200)
    for k in range(nd): net.classify_window(W[k % len(W)], stats=st)
    res["ops_per_decision"] = op_table(net, st, nd)
    if a.idle_seconds > 0 and a.active_seconds > 0:
        ph = {}; ph["idle_start"] = time.time(); time.sleep(a.idle_seconds); ph["idle_end"] = time.time()
        if gpio: gpio.output(a.gpio_pin, 1)
        ph["active_start"] = time.time(); n_act = 0
        while time.time() - ph["active_start"] < a.active_seconds: net.classify_window(W[n_act % len(W)]); n_act += 1
        ph["active_end"] = time.time()
        if gpio: gpio.output(a.gpio_pin, 0)
        res["phases"] = ph; res["active_decisions"] = n_act
    res["rss_bytes"]["peak"] = rss_bytes(); res["board_after"] = board_state()
    json.dump(res, open(a.out, "w"), indent=1)
    L = res["latency_ms"]["decision_L1_full_pipeline"]; o = res["ops_per_decision"]
    print(f"decision latency ms: p50 {L['p50']:.2f} p95 {L['p95']:.2f} p99 {L['p99']:.2f} (n={L['n']}) | front end p50 {res['latency_ms']['front_end_only']['p50']:.2f}"
          f" | peak RSS {res['rss_bytes']['peak']/2**20:.1f} MiB (after imports {rss_imports/2**20:.1f}) | artifact {res['storage']['artifact_bytes']} B, {params} params"
          f" | SynOps/decision {o['synops_total_proxy']:.0f} vs dense-equivalent MACs {o['dense_equivalent_macs_total']:.0f}")

if __name__ == "__main__": main()
