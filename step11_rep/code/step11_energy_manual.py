#!/usr/bin/env python3
"""Energy per decision with a USB inline meter that only DISPLAYS an energy counter (no logging), e.g. a USB-C PD tester.
NumPy + stdlib only. One run = one idle phase + one active phase; repeat >= 3 times.

  python3 energy_manual.py snn_bio_model.npz bench_windows.npy --seconds 600 --out energy_run1.json [--unit mWh|Wh]

You will be asked to: reset the meter's energy counter and press Enter (starts a phase); when a phase ends, press Enter at
the moment you READ the counter (this timestamp is what counts), then type the reading.
Idle power      P_idle = E_idle / (time from reset-Enter to read-Enter)
Active phase    the counter also runs during the idle tail after the loop ends, so
                E_incremental = E_active - P_idle * (time from reset-Enter to read-Enter)
Per decision    E_incremental / decisions  (a decision = 5 windows, mean log-probability, as in the bundle).
This is whole-board energy above idle; meter accuracy and resolution limit it (use long phases, e.g. 600 s).
"""
import argparse, json, platform, sys, time
import numpy as np
from step11_rpi_infer import WaveSpikeBioNP


def ask_mark(msg):
    input(msg); return time.time()


def ask_value(msg, unit):
    while True:
        s = input(msg).strip().replace(",", ".")
        try:
            v = float(s); return v*(1000.0 if unit == "Wh" else 1.0)          # always stored as mWh
        except ValueError:
            print("  not a number, try again")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("windows")
    ap.add_argument("--seconds", type=float, default=600); ap.add_argument("--out", default="energy_run.json")
    ap.add_argument("--unit", choices=("mWh", "Wh"), default="mWh")
    a = ap.parse_args()
    net = WaveSpikeBioNP(a.model); W = np.load(a.windows).astype(np.float32); n_agg = net.n_agg or 5
    groups = [W[i:i + n_agg] for i in range(0, len(W) - n_agg + 1, n_agg)]
    for g in groups[:2]: net.decide(g)                                          # warm-up
    print(f"model {a.model} | {len(groups)} distinct decisions of {n_agg} windows | phases of {a.seconds:.0f} s | meter unit {a.unit}")

    t0 = ask_mark("\n[IDLE] Reset the meter's energy counter, then press Enter (keep the Pi otherwise idle) ")
    time.sleep(a.seconds); print("[IDLE] done.")
    t1 = ask_mark("[IDLE] Press Enter at the moment you read the energy counter ")
    e_idle = ask_value(f"[IDLE] Type the reading in {a.unit}: ", a.unit)

    t2 = ask_mark("\n[ACTIVE] Reset the meter's energy counter, then press Enter ")
    ta = time.time(); n = 0; lat = []
    while time.time() - ta < a.seconds:
        s = time.perf_counter(); net.decide(groups[n % len(groups)]); lat.append(time.perf_counter() - s); n += 1
    tb = time.time(); print(f"[ACTIVE] done: {n} decisions in {tb - ta:.1f} s.")
    t3 = ask_mark("[ACTIVE] Press Enter at the moment you read the energy counter ")
    e_act = ask_value(f"[ACTIVE] Type the reading in {a.unit}: ", a.unit)

    T_idle, T_act_counted = t1 - t0, t3 - t2
    p_idle_w = e_idle*3.6/T_idle                                               # mWh -> J (x3.6), / s -> W
    e_incr_j = e_act*3.6 - p_idle_w*T_act_counted
    res = {"model": a.model, "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
           "machine": platform.machine(), "seconds_requested": a.seconds, "meter_unit_entered": a.unit,
           "idle": {"reading_mWh": e_idle, "counted_seconds": T_idle, "P_idle_W": p_idle_w},
           "active": {"reading_mWh": e_act, "counted_seconds": T_act_counted, "loop_seconds": tb - ta, "decisions": n,
                      "ms_per_decision_mean": 1e3*float(np.mean(lat)), "ms_per_decision_p95": 1e3*float(np.percentile(lat, 95))},
           "E_incremental_J": e_incr_j, "E_incremental_mJ_per_decision": 1e3*e_incr_j/n,
           "P_active_above_idle_W": e_incr_j/(tb - ta),
           "note": "whole-board energy above idle from a display-only meter; reaction time and meter resolution limit accuracy"}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nP_idle {p_idle_w:.3f} W | active +{res['P_active_above_idle_W']:.3f} W above idle | "
          f"{res['E_incremental_mJ_per_decision']:.2f} mJ per decision (incremental) | {res['active']['ms_per_decision_mean']:.1f} ms per decision"
          f" | saved {a.out}")
    if e_incr_j <= 0: print("WARNING: incremental energy <= 0; phases too short for the meter's resolution, or the readings were swapped.")


if __name__ == "__main__":
    main()
