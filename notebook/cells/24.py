# §9b Run the ablation (resumable; stops launching new runs after MAX_SESSION_H hours)
# PARALLEL (env WAVESPIKE_PARALLEL=1, off by default): several processes run this cell at once on the same RES_DIR.
# Each run is claimed with an exclusive lock file first; a lock whose process is gone (same host) is taken over.
# Runs are still visited in repeat -> fold -> variant order, so the finished set stays close to complete repeats.
PARALLEL = os.environ.get("WAVESPIKE_PARALLEL") == "1"
LOCK_DIR = f"{RES_DIR}/locks"
import socket
_HOST = socket.gethostname()

def _claim(p):
    if not PARALLEL: return True
    os.makedirs(LOCK_DIR, exist_ok=True); lk = f"{LOCK_DIR}/{os.path.basename(p)}.lock"
    for _ in range(2):
        try:
            fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, f"{_HOST} {os.getpid()}".encode()); os.close(fd)
            if os.path.exists(p): os.remove(lk); return False         # finished by another worker in the meantime
            return True
        except FileExistsError:
            try: host, pid = open(lk).read().split()
            except (OSError, ValueError): return False                # being written right now
            if host != _HOST: return False
            try: os.kill(int(pid), 0); return False                   # owner still alive
            except ProcessLookupError: os.remove(lk)                  # stale: owner died; retry once
            except PermissionError: return False
    return False

def _release(p):
    if PARALLEL:
        try: os.remove(f"{LOCK_DIR}/{os.path.basename(p)}.lock")
        except FileNotFoundError: pass

if PARALLEL: print(f"PARALLEL worker pid {os.getpid()} on {_HOST}: claiming runs via {LOCK_DIR}")
_stop = False
for cs, ts in REPEATS:
    for k in range(CFG.n_folds):
        for v in RUN_VARIANTS:
            p = run_path(v, cs, ts, k)
            if os.path.exists(p): continue
            if (time.time() - NOTEBOOK_T0)/3600 > MAX_SESSION_H:
                print(f"⏸ session budget of {MAX_SESSION_H} h reached; run the notebook again to resume."); _stop = True; break
            if not _claim(p): continue
            try: r = run_one(v, cs, ts, k)
            finally: _release(p)
            print(f"{v:14s} cv{cs} ts{ts} fold {k}: AUROC best {r['best']['auroc']:.3f} / last {r['last']['auroc']:.3f} | "
                  f"epochs {r['epochs_run']} (best {r['best_epoch']}) | {r['minutes']:.1f} min", flush=True)
        if _stop: break
    if _stop: break
