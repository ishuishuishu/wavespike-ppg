# §1 Setup
import os, sys, subprocess
_CUBLAS_PRESET = "CUBLAS_WORKSPACE_CONFIG" in os.environ
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # P0.1: must be set before CUDA initialises
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "wfdb", "synapseclient"], check=False)
_TORCH_WAS_LOADED = "torch" in sys.modules

import math, time, json, random, copy, glob, zipfile, urllib.request, urllib.parse, re, shutil, hashlib, tempfile, importlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict, field
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.signal import resample_poly, find_peaks, butter, sosfiltfilt
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

NOTEBOOK_T0 = time.time()
SMOKE = os.environ.get("WAVESPIKE_SMOKE") == "1"     # local CPU test of the notebook only; never set in Colab
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False      # fp32 everywhere, so the GPU and NumPy (Pi) models agree
torch.backends.cudnn.allow_tf32 = False

# P0.1 determinism
DETERMINISTIC = True
torch.backends.cudnn.deterministic = DETERMINISTIC
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(DETERMINISTIC)
if DEVICE.type == "cuda" and _TORCH_WAS_LOADED and not _CUBLAS_PRESET and torch.cuda.is_initialized():
    print("⚠ CUDA was initialised before CUBLAS_WORKSPACE_CONFIG was set. Use Runtime → Restart session, then Run all.")
print("torch", torch.__version__, "| device:", DEVICE, "| deterministic:", DETERMINISTIC,
      "| CUBLAS_WORKSPACE_CONFIG =", os.environ["CUBLAS_WORKSPACE_CONFIG"])

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class StopForUser(RuntimeError):
    """Raised on purpose: the notebook needs a decision from you. Read the message, edit §2, run again."""
