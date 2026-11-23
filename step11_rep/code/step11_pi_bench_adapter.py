"""Installed in the Pi bundle as `rpi_infer.py`, so the teammate's edge/pi_bench.py (copied unchanged) benchmarks the
Step 11 model. pi_bench.py does `import rpi_infer`, builds rpi_infer.WaveSpikeNP(path), and calls .encoder_input(x),
.classify_window(x) and .classify_window(x, stats=dict); it also reads .W_in, .W_rec and .W_out.

WaveSpikeNP here is step11_rpi_infer.WaveSpikeBioNP (amplitude fusion included), with the teammate's return signature
(class, probs, hidden spike rate, logits). Note: at teammate commit d2c7fe0, their own rpi_infer.WaveSpikeNP.classify_window
takes no `stats` argument, so their pi_bench.py stops with a TypeError at its op-count step. This adapter accepts it.
Without `stats`, no diagnostics are computed (so latency is not inflated) and the hidden rate is returned as nan.
"""
import numpy as np
from step11_rpi_infer import WaveSpikeBioNP


class WaveSpikeNP(WaveSpikeBioNP):
    def classify_window(self, x, stats=None):
        if stats is None:
            lg = self.window_logits(x); rate = float("nan")
        else:
            h0, s0 = stats.get("hid_spikes", 0), stats.get("steps", 0)
            lg = self.window_logits(x, stats)
            rate = (stats["hid_spikes"] - h0)/max(1, (stats["steps"] - s0)*self.W_rec.shape[0])
        p = self.softmax(lg.astype(np.float64))
        return self.classes[int(p.argmax())], p, rate, lg
