# CHANGES: run the handoff folders on a GPU

**For the teammate:** paste this whole file into Claude Code, opened in the repository folder, and say: *"Apply these changes exactly as written, then run the checks in section 4."* It applies to whichever handoff folders you have: `lean_snn/`, `paper_designs/`, `ablation/`, `p6/`.

**For Claude applying this:**
- Make **only** the edits listed here.
- Each edit gives the exact text to find and the exact text to replace it with.
- If a "find" text isn't in a file, skip that file and say so. Don't improvise a different edit.
- Don't change any other setting, default, variant or line of code.

---

## 1. Why these changes

The handoff guides were written for a CPU, because that's how our own runs were made. The training code already uses an NVIDIA GPU automatically when PyTorch can see one (`code/cells/02.py`: `DEVICE = cuda if available, else cpu`), and it was used on a Colab GPU before. Two things still stop or slow a GPU run, and the guides tell you to install the CPU-only PyTorch:

| # | Problem | Affects | Fix |
|---|---|---|---|
| A | Determinism is set to "error" mode. If one GPU operation has no deterministic version, PyTorch stops the run with a `RuntimeError` | Every folder | On GPU, warn instead of stop (§2.A) |
| B | The P1 (DoE) front end builds a tensor on the CPU inside the forward pass, which crashes when the model is on the GPU (`Expected all tensors to be on the same device`) | Only the P1 variants (`x_p1_*`). No handoff runs P1, but the file is shared, so fix it everywhere for safety | Create it on the same device (§2.B) |
| C | The READMEs install CPU-only PyTorch and say "CPU only" / "A GPU isn't needed or used" | Every folder | §3 |

Nothing else changes: same variants, folds, seeds, training recipe (150 epochs, 1,600 windows per epoch, batch 64), metrics and saved-model format. Saved models load on CPU or GPU.

---

## 2. Code changes (in each folder's `code/`)

### 2.A Determinism: warn instead of stop on GPU

**Files:** `lean_snn/code/cells/02.py`, `paper_designs/code/cells/02.py`, `ablation/code/cells/02.py`, `p6/code/cells/02.py`

Find (line 39):
```python
torch.use_deterministic_algorithms(DETERMINISTIC)
```
Replace with:
```python
torch.use_deterministic_algorithms(DETERMINISTIC, warn_only=(DEVICE.type == "cuda"))   # GPU: warn, don't stop, if an op has no deterministic CUDA version
```

- On CPU nothing changes: it still stops on non-deterministic operations, as before.
- On GPU a warning is printed instead. Copy any such warning into your report.

### 2.B P1 front end: build the kernel on the model's device

**Files:** `lean_snn/code/spiking_wavelets.py`, `paper_designs/code/spiking_wavelets.py`, `ablation/code/spiking_wavelets.py`, `p6/code/spiking_wavelets.py`

Find (inside `def li_fir(mu, dt, n):`):
```python
        t = torch.arange(n, dtype=torch.float32)
```
Replace with:
```python
        t = torch.arange(n, dtype=torch.float32, device=mu.device)          # same device as mu (GPU-safe)
```

`mu` is always a tensor at both places that call `li_fir`, so this is safe on CPU too. The result is identical on CPU.

---

## 3. README changes (in each folder's `README.md`)

### 3.A Install GPU PyTorch instead of CPU PyTorch

**Files:** all four `README.md`

Find:
```text
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Replace with:
```text
pip install torch      # with an NVIDIA GPU: use the exact command from pytorch.org (Get Started) for your CUDA version
```

### 3.B Python/CPU line

**Files:** `paper_designs/README.md`, `ablation/README.md`, `p6/README.md`

Find:
```text
- Python 3.13 or newer, CPU only. Our runs used Python 3.14 and PyTorch 2.15.
```
Replace with:
```text
- Python 3.13 or newer. GPU (NVIDIA, CUDA PyTorch) or CPU both work: the code uses the GPU automatically if PyTorch can see one. Our runs used Python 3.14 and PyTorch 2.15 on CPU.
- Check the GPU is visible: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"` must print `True` and the GPU name. Every run also prints `device: cuda` at the start.
```

**File:** `lean_snn/README.md`

Find:
```text
2. Create a Python 3.13 (or newer) environment with CPU PyTorch and the packages in `requirements.txt`. A GPU isn't needed or used. The original runs used Python 3.14 and PyTorch 2.15.
```
Replace with:
```text
2. Create a Python 3.13 (or newer) environment with PyTorch (CUDA version if you have an NVIDIA GPU; the code uses it automatically) and the packages in `requirements.txt`. The original runs used Python 3.14 and PyTorch 2.15 on CPU. Check the GPU is visible: `python -c "import torch; print(torch.cuda.is_available())"` must print `True`, and every run prints `device: cuda` at the start.
```

### 3.C "One job at a time / 12 threads" rule

**Files:** all four `README.md`

Find:
```text
- Run one job at a time. About 12 threads is fastest on a CPU.
```
Replace with:
```text
- On CPU: run one job at a time; about 12 threads is fastest.
- On GPU: the thread number only affects the CPU side and can stay at the default. You may run 2-4 **different** jobs at once on one GPU if memory allows (watch `nvidia-smi`). Never run the same variant on the same dataset twice at the same time.
```

### 3.D Record the device in `versions.txt`

**Files:** `paper_designs/README.md`, `ablation/README.md`, `p6/README.md`

Find:
```text
  python -c "import sys, torch, numpy; print(sys.version, torch.__version__, numpy.__version__)"; } > versions.txt
```
Replace with:
```text
  python -c "import sys, torch, numpy; print(sys.version, torch.__version__, numpy.__version__, 'cuda' if torch.cuda.is_available() else 'cpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"; } > versions.txt
```

**File:** `lean_snn/README.md`

Find:
```text
   python -c "import sys, torch, numpy; print(sys.version, torch.__version__, numpy.__version__)"
```
Replace with:
```text
   python -c "import sys, torch, numpy; print(sys.version, torch.__version__, numpy.__version__, 'cuda' if torch.cuda.is_available() else 'cpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### 3.E Thread wording in timing lines (optional, wording only)

Lines like "about 6 min each on 12 threads", or "Time (12 CPU threads)", are **CPU timings from our machine**. Leave them, and add after the first one in each README:
```text
(These are CPU timings. On a GPU, transformer designs are usually much faster; our small recurrent SNN steps through 500 time points one by one, so it may gain less.)
```

---

## 4. Checks after applying (Claude: run these and report the output)

Run from inside each folder you changed, with the environment active and `WAVESPIKE_DATA_ROOT` set.

1. **The edits are in:**
   ```bash
   grep -n "warn_only" code/cells/02.py
   grep -n "device=mu.device" code/spiking_wavelets.py
   ```
   Each must print one line.
2. **GPU visible:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
3. **Smoke test on GPU.** It writes no result files. It must print `device: cuda`, then one result line, with no `Traceback`:
   ```bash
   python -I code/bio_run.py --dataset zenodo_bio --variants x_tau_amp --folds 0 --seeds 0 --smoke
   ```
   In `paper_designs/`, also smoke-test one transformer:
   ```bash
   python -I code/bio_run.py --dataset zenodo_bio --variants x_stf_swsa --folds 0 --seeds 0 --smoke
   ```
4. **Any determinism warnings?** If step 3 printed a `UserWarning` mentioning "deterministic", copy it into your report.
5. **A saved model still loads**, if you have one:
   ```bash
   bash check_models.sh
   ```
   Each must end in `runs OK`.

If any step fails, stop and send the output to the project lead. Don't change other code to make it pass.

---

## 5. What this means for the results (tell the project lead)

- **The same model on GPU and CPU gives slightly different numbers.** The arithmetic rounds differently, and spiking networks amplify tiny differences (a spike fires or doesn't). The code already uses full fp32 precision (TF32 off) to keep this small. Expect run-to-run differences similar to changing the seed, not systematic shifts.
- **Your runs are combined with ours in some folders:**
  - `p6`: your folds 1-2 plus our CPU fold 0;
  - `ablation`: your runs plus our CPU runs;
  - `lean_snn` task A: your seed 1 plus our CPU seed 0.

  That's fine, but **say in your note that you used a GPU**, and include `versions.txt`. We'll mark these rows as mixed-device.
- `paper_designs` is all new runs, so all of its numbers will be from your GPU.
- **Speed:** the transformer designs (`x_stf_*`) should be much faster on a GPU. Our recurrent SNN and the P2/P4/P6 front-end designs loop over 500 time steps with small batches, so the GPU gain may be modest. Running 2-4 different jobs at once on the GPU (§3.C) is the easiest way to finish sooner.

**Not tested on a GPU by us:** our workstation has no GPU. The changes were checked by reading the code. The checks in §4 are how you confirm they work on your machine.
