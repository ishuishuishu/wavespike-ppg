# Execute the notebook's code cells in order in one namespace (local CPU smoke test). Usage: run_test.py NB [stop_after_cell_id]
import json, sys, os, time, traceback
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; plt.show = lambda *a, **k: plt.close("all")
nb = json.load(open(sys.argv[1])); stop = sys.argv[2] if len(sys.argv) > 2 else None
sys.path.insert(0, os.getcwd()); ns = {"__name__": "__main__"}
for c in nb["cells"]:
    if c["cell_type"] != "code": continue
    t0 = time.time(); print(f"\n######## cell {c['id']} ########", flush=True)
    try:
        exec(compile("".join(c["source"]), f"<cell {c['id']}>", "exec"), ns)
    except Exception as e:
        if type(e).__name__ == "StopForUser":
            print(f"######## StopForUser in {c['id']}: {e}"); sys.exit(3)
        traceback.print_exc(); sys.exit(1)
    print(f"######## cell {c['id']} ok ({time.time()-t0:.1f}s)", flush=True)
    if stop and c["id"] == stop: break
print("ALL CELLS OK")
