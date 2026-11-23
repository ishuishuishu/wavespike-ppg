#!/usr/bin/env python3
"""Load a model saved with bio_run.py --save-model and check it rebuilds and runs.

Usage: python -I lean_snn/code/load_model.py <model.pt> [best|last]      (default: last, the checkpoint we report)

It rebuilds the network from the saved config, loads the weights (strict: every tensor must match), runs it on
one synthetic 10 s window and prints the class probabilities. This is the starting point for the Raspberry Pi
port; it needs PyTorch. The NumPy-only Pi exporter (notebook cell 28) does not yet cover the amplitude-fusion term.
"""
import os, sys, tempfile
import numpy as np, torch

# the shared notebook code creates raw/processed/results under the data root; no data is read here
os.environ.setdefault("WAVESPIKE_DATA_ROOT", os.path.join(tempfile.gettempdir(), "wavespike_data"))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                           # bio_run.py and spiking_wavelets.py sit next to this file
import bio_run                                                     # noqa: E402
from spiking_wavelets import make_paper_frontends, make_token_nets  # noqa: E402


def load(path, which="last"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    g = bio_run.load_notebook_code(); g["VARIANTS"].update(bio_run.BIO_VARIANTS)
    make_token_nets(g); bio_run.make_amp_fusion(g); bio_run.make_packet(g); make_paper_frontends(g)
    cfg = g["Config"]()
    for k, v in ck["cfg"].items(): setattr(cfg, k, v)
    g["AMP_COLS"] = len(ck["amp_norm"]["mean"]) if ck["amp_norm"] else 0
    m = g["build_model"](cfg); m.load_state_dict(ck[f"{which}_state"], strict=True); m.eval()
    return m, cfg, ck


if __name__ == "__main__":
    path = sys.argv[1]; which = sys.argv[2] if len(sys.argv) > 2 else "last"
    m, cfg, ck = load(path, which)
    print(f"{ck['variant']} | {ck['dataset']} fold {ck['fold']} seed {ck['seed']} | {which} weights | "
          f"{sum(p.numel() for p in m.parameters() if p.requires_grad)} parameters | {len(ck['identities'])} identities")
    n = int(cfg.win_sec * cfg.fs); t = np.arange(n) / cfg.fs
    x = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t)          # synthetic pulse, ~72 bpm
    x = ((x - x.mean()) / (x.std() + 1e-6)).astype(np.float32)[None]           # same per-window z-score as training
    if ck["amp_norm"]: x = np.concatenate([x, np.zeros((1, len(ck["amp_norm"]["mean"])), np.float32)], 1)
    with torch.no_grad(): lg, _ = m(torch.tensor(x))
    p = torch.softmax(lg, 1)[0].numpy()
    print("runs OK | predicted identity", ck["identities"][int(p.argmax())], "| top probability", round(float(p.max()), 3))
