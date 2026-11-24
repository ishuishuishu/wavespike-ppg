# Raspberry Pi 3 benchmark (our measurements)

**Board:** Raspberry Pi 3 Model B rev 1.2: 4× ARM Cortex-A53 at 1.2 GHz, 1 GB RAM, 64-bit Raspberry Pi OS (Linux 6.18, aarch64), Python 3.13.5, NumPy 2.5.5.

**Method:** `edge/pi_bench.py` with `edge/rpi_infer.py`, NumPy only:
- one core (`taskset -c 0`), one thread;
- 100 untimed warm-up windows, then 500 timed decisions per run;
- 5 runs per model, alternating models.

**Inputs:** real test windows of each model's own split (split 1, seed 0). Before benchmarking, `edge/pi_check.py` confirmed the Pi gives exactly the expected top-1 on all four models.

Per-run raw numbers: `pi_results_raspi3.csv`.

| Model | p50 latency per 10 s window (mean ± SD of 5 runs) | p95 | p99 | Peak RAM | Model file | Faster than real time |
|---|---|---|---|---|---|---|
| Best SNN (`x_aug_tau_amp`), Zenodo | **185.1 ± 0.7 ms** | 189.6 ms | 205.8 ms | 30.4 MiB | 33 KB | 54× |
| Best SNN, Seeing Red | **186.8 ± 1.4 ms** | 192.1 ms | 208.5 ms | 30.4 MiB | 33 KB | 54× |
| Lean SNN (`x_aug_lean_tau_amp`), Zenodo | **93.9 ± 0.4 ms** | 95.6 ms | 108.2 ms | 30.4 MiB | 32 KB | 106× |
| Lean SNN, Seeing Red | **93.9 ± 0.7 ms** | 96.1 ms | 108.4 ms | 30.4 MiB | 32 KB | 106× |

**Spiking activity on these windows:**

| Model | Hidden spike rate | Input slots silent | Spike operations per window | Dense multiply-adds per window | Ratio |
|---|---|---|---|---|---|
| Best, Zenodo | 3.8% | 96.6% | 112k | 2.98 M | 27× fewer |
| Lean, Zenodo | 4.8% | 95.5% | 68k | 1.42 M | 21× fewer |
| Best, Seeing Red | 6.9% | 92.3% | 202k | 2.88 M | 14× fewer |
| Lean, Seeing Red | 10.2% | 91.4% | 138k | 1.38 M | 10× fewer |

- **Lean vs best:** 1.97–1.99× faster on the same board.
- **Wavelet front end:** 2.6–3.1 ms, under 2% of the decision time.
- **Board condition:** clock at 1.2 GHz throughout, CPU 55.8–69.8 °C (55.8 °C was the cool start of the very first run), **no throttling in any run** (`throttled=0x0`). Run-to-run spread under 1%.
- **Why latency follows the dense workload:** on a CPU, NumPy still computes the recurrent layer as a dense matrix product. The spike-operation counts show what event-driven hardware would need.
- **5-window decision** (about 50 s of pulse): about 5 × the per-window time, i.e. 0.93 s (best) or 0.47 s (lean).

**Energy:** not measured; no power meter was connected. An estimate is latency × the Pi 3's extra power with one core busy (to be cited from a published measurement). Measured energy needs a USB power meter (`PI_RUN.md` §5).
