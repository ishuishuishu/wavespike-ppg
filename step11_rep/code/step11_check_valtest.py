#!/usr/bin/env python3
"""Step 11 check: Part B (enlarged pool) validation/test windows are identical, by content and order, to Part A's;
extra windows come only from (person, day) pairs that are training days in that fold."""
import numpy as np, os, sys
sys.path.insert(0, os.path.expanduser("~/capstone_metrics_run/wavespike-ppg/biometrics")); import bio_run as B
r = os.environ["WAVESPIKE_DATA_ROOT"] + "/processed/"
a = np.load(r + "zenodo_bio_50hz_10s.npz", allow_pickle=True); b = np.load(r + "step11_zenodo_more_50hz_10s.npz", allow_pickle=True)
ex = b["is_extra"]; fa = B.session_folds(a["y"], a["sess"]); fb = B.session_folds(b["y"], b["sess"], E=~ex)
ok = True
for k in range(3):
    for j, nm in ((1, "val"), (2, "test")):
        ia, ib = fa[k][j], fb[k][j]
        same = len(ia) == len(ib) and all(np.array_equal(a[key][ia], b[key][ib]) for key in ("X", "y", "vid", "sess"))
        ok &= same; print(f"fold {k} {nm}: {len(ia)} vs {len(ib)} windows, identical content and order: {same}")
    ta, tb = fa[k][0], fb[k][0]; tbo = tb[~ex[tb]]; tbe = tb[ex[tb]]
    cont = np.array_equal(a["X"][ta], b["X"][tbo]); ok &= cont
    days_tr = set(zip(a["y"][ta].tolist(), a["sess"][ta].tolist())); days_ex = set(zip(b["y"][tbe].tolist(), b["sess"][tbe].tolist()))
    ok &= days_ex <= days_tr
    print(f"fold {k} train: base {len(ta)} | more {len(tb)} ({len(tbe)} extra) | base train == more non-extra train: {cont} | "
          f"extra only from training days: {days_ex <= days_tr}")
print("VALTEST_IDENTICAL" if ok else "VALTEST_DIFFER")
