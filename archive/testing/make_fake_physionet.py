# Build a FAKE local copy of the PhysioNet/Zenodo layouts used by the notebook, for the local pipeline test only.
# The column names below are invented test values; the real ones must be read from the real metadata.csv.
import os, sys, zipfile, numpy as np, pandas as pd, wfdb
root, data_root = sys.argv[1], sys.argv[2]; ALL_IDS = "--all-ids" in sys.argv
proj = os.path.join(root, "mimic-iii-ext-ppg", "1.1.0"); os.makedirs(proj, exist_ok=True)
rng = np.random.default_rng(1)

def ppg(sec, fs, af, hr=75):
    t, out = 0, []
    while len(out) < sec*fs:
        rr = 60/hr*(np.clip(1 + 0.22*rng.standard_normal(), .5, 1.7) if af else 1 + 0.02*rng.standard_normal())
        n = int(rr*fs); tt = np.linspace(0, 1, n, endpoint=False)
        out += list(np.exp(-0.5*((tt-.25)/.08)**2) + .4*np.exp(-0.5*((tt-.55)/.1)**2))
    x = np.array(out[:sec*fs]); return x + 0.02*rng.standard_normal(len(x))

rows = []
for s in range(1, 41):
    sid = 100 + s; has_af = s % 2 == 0
    labs = (["AF"]*6 + ["SR"]*5 + ["AFL", "STACH"]) if has_af else (["SR"]*8 + ["SBRAD", "STACH"])
    for j, lab in enumerate(labs):
        rel = f"p{sid//100:02d}/p{sid:06d}/seg_{j:04d}"; d = os.path.join(proj, os.path.dirname(rel)); os.makedirs(d, exist_ok=True)
        x = ppg(30, 125, lab == "AF")
        if j == 0 and s % 5 == 0: x[100:110] = np.nan          # NaN segment -> dropped
        if j == 1 and s % 7 == 0: x[:] = 0.5                   # flat -> dropped
        if j == 2 and s % 9 == 0: x[500:500+125*7] = x[500]    # 7 s flat run -> <25 s valid -> dropped
        if j == 3 and s % 11 == 0: x = x[:125*27]              # 27 s segment -> kept, edge-padded
        sig = np.stack([np.sin(np.arange(len(x))/10), x], 1)
        wfdb.wrsamp(os.path.basename(rel), fs=125, units=["mV", "NU"], sig_name=["II", "PLETH"], p_signal=sig,
                    fmt=["16", "16"], write_dir=d)
        rows.append({"subject_id": f"{sid}", "rhythm": lab, "record_path": rel + ".hea", "hr": 75})
pd.DataFrame(rows).to_csv(os.path.join(proj, "metadata.csv"), index=False)
open(os.path.join(proj, "README.md"), "w").write("FAKE README for the local test. Columns: subject_id (MIMIC-III subject), rhythm "
                                                 "(AF, SR, AFL, STACH, SBRAD), record_path (WFDB header path), hr.\n")
# fake MIMIC PERform AF zips (pre-placed on the 'Drive' so nothing is downloaded); IDs 102 and 104 overlap with Ext-PPG
raw = os.path.join(data_root, "raw", "mimic_perform_af"); os.makedirs(raw, exist_ok=True)
for name, af, ids in [("mimic_perform_af_wfdb", True, [102, 104, 900]), ("mimic_perform_non_af_wfdb", False, [901, 902])]:
    tmp = os.path.join(root, name); os.makedirs(tmp, exist_ok=True)
    for i, sid in enumerate(ids):
        x = ppg(180, 125, af)
        wfdb.wrsamp(f"{name[:-5]}_{i+1:03d}", fs=125, units=["NU"], sig_name=["PPG"], p_signal=x[:, None], fmt=["16"],
                    write_dir=tmp, comments=[f"<subject_id>: p{sid:06d}" if (sid != 902 or ALL_IDS) else "no id here"])
    with zipfile.ZipFile(os.path.join(raw, name + ".zip"), "w") as z:
        for f in os.listdir(tmp): z.write(os.path.join(tmp, f), os.path.join(name, f))
# fake CinC 2015 (training/RECORDS + headers with alarm comments; 250 Hz, 300 s; some without PLETH)
cdir = os.path.join(root, "challenge-2015", "1.0.0", "training"); os.makedirs(cdir, exist_ok=True); names = []
for i in range(24):
    kind = ["t", "v", "a"][i % 3]; truth = i % 2 == 0; name = f"{kind}{100+i}s"; names.append(name)
    x = ppg(300, 250, af=not truth, hr=130); chans = ["II", "PLETH"] if i % 6 != 5 else ["II", "V"]
    wfdb.wrsamp(name, fs=250, units=["mV", "NU"], sig_name=chans, p_signal=np.stack([np.sin(np.arange(len(x))/9), x], 1),
                fmt=["16", "16"], write_dir=cdir,
                comments=[{"t": "Tachycardia", "v": "Ventricular_Tachycardia", "a": "Asystole"}[kind], "True alarm" if truth else "False alarm"])
open(os.path.join(cdir, "RECORDS"), "w").write("\n".join(names) + "\n")
print("fake PhysioNet at", root, "|", len(rows), "segments")
