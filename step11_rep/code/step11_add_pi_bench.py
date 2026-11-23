#!/usr/bin/env python3
"""Step 11 Part C, last protocol item: make the Pi bundle runnable with the teammate's edge/pi_bench.py (unchanged),
and add a timed energy mode for a display-only USB meter. Only ADDS files to the bundle; never overwrites.

Usage: python -I step11_add_pi_bench.py <bundle dir>
Adds: pi_bench.py (teammate's, byte-identical), rpi_infer.py (adapter = step11_pi_bench_adapter.py),
      energy_manual.py (= step11_energy_manual.py), bench_windows.npy (the 200 raw test windows already in samples.npz),
      run_pi_bench.sh, PI_BENCH.md
"""
import hashlib, os, shutil, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("WAVESPIKE_REPO", os.path.join(os.path.expanduser("~"), "capstone_metrics_run", "wavespike-ppg"))


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    b = sys.argv[1]
    new = ["pi_bench.py", "rpi_infer.py", "energy_manual.py", "bench_windows.npy", "run_pi_bench.sh", "PI_BENCH.md"]
    clash = [f for f in new if os.path.exists(os.path.join(b, f))]
    assert not clash, f"refusing to overwrite {clash}"
    for f in ("snn_bio_model.npz", "step11_rpi_infer.py", "samples.npz"): assert os.path.exists(os.path.join(b, f)), f
    src_bench = os.path.join(REPO, "edge", "pi_bench.py")
    shutil.copy(src_bench, os.path.join(b, "pi_bench.py")); assert sha(src_bench) == sha(os.path.join(b, "pi_bench.py"))
    shutil.copy(os.path.join(HERE, "step11_pi_bench_adapter.py"), os.path.join(b, "rpi_infer.py"))
    shutil.copy(os.path.join(HERE, "step11_energy_manual.py"), os.path.join(b, "energy_manual.py"))
    W = np.load(os.path.join(b, "samples.npz"))["windows"].astype(np.float32); np.save(os.path.join(b, "bench_windows.npy"), W)
    with open(os.path.join(b, "run_pi_bench.sh"), "w") as fh:
        fh.write("#!/usr/bin/env bash\n# Teammate's pi_bench.py on the Step 11 model: 5 repetitions, one core, one thread.\n"
                 "set -euo pipefail\ncd \"$(dirname \"$0\")\"\n"
                 "PIN=\"\"; command -v taskset >/dev/null && PIN=\"taskset -c 0\"\n"
                 "for r in 1 2 3 4 5; do OMP_NUM_THREADS=1 $PIN python3 pi_bench.py --model snn_bio_model.npz "
                 "--windows bench_windows.npy --n 1000 --warmup 100 --out bench_run$r.json; done\n")
    os.chmod(os.path.join(b, "run_pi_bench.sh"), 0o755)
    with open(os.path.join(b, "PI_BENCH.md"), "w") as fh:
        fh.write(f"""# Benchmarking on the Raspberry Pi

Run `run_pi_selfcheck.sh` first (correctness). Then:

**Latency, memory, op counts:** `./run_pi_bench.sh`, which runs the teammate's `edge/pi_bench.py` unchanged
(SHA-256 `{sha(src_bench)}`) 5 times and writes `bench_run1..5.json`.
- `pi_bench.py` imports `rpi_infer`. In this bundle `rpi_infer.py` is a small adapter around `step11_rpi_infer.py`, so the
  amplitude-fusion head is included. It is needed because, at teammate commit d2c7fe0, their own `rpi_infer.py` has no `stats`
  argument and `pi_bench.py` stops with a TypeError.
- `pi_bench.py` times **one 10 s window**. A verification decision uses 5 windows, so multiply by 5 (or use `energy_manual.py`,
  which times whole decisions).
- Its built-in energy mode (`--idle-seconds/--active-seconds`, `--energy-from --power-csv`) needs a meter that LOGS voltage and current.

**Energy with a display-only USB meter** (e.g. a USB-C PD tester with an mWh counter):
1. Put the meter between the power supply and the Pi (a Pi 3B needs a USB-C to micro-USB cable after the meter). Use a fixed CPU governor
   (`echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`), and turn Wi-Fi and Bluetooth off if possible.
2. `python3 energy_manual.py snn_bio_model.npz bench_windows.npy --seconds 600 --out energy_run1.json` and follow the prompts.
   Press Enter at the moment you read the counter, then type the value.
3. Repeat 3 times (energy_run2, energy_run3); report mean ± SD of `E_incremental_mJ_per_decision` and the meter model and accuracy.

Do not run python with `-I` here: `pi_bench.py` and `energy_manual.py` import modules from this directory.
""")
    print("added:", ", ".join(new))


if __name__ == "__main__":
    main()
