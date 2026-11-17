# Assemble wavespike_ppg_colab_v3.ipynb from cells/NN.{py,md} (in name order).
import json, glob, os, sys
here = os.path.dirname(os.path.abspath(__file__))
files = sorted(f for f in glob.glob(f"{here}/cells/[0-9][0-9].*"))
cells = []
for f in files:
    src = open(f).read().rstrip("\n")
    lines = [l + "\n" for l in src.split("\n")]; lines[-1] = lines[-1].rstrip("\n")
    cid = "c" + os.path.basename(f).split(".")[0]
    if f.endswith(".md"): cells.append({"cell_type": "markdown", "id": cid, "metadata": {}, "source": lines})
    else: cells.append({"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None, "outputs": [], "source": lines})
nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}}
out = sys.argv[1] if len(sys.argv) > 1 else f"{here}/wavespike_ppg_colab_v3.ipynb"
json.dump(nb, open(out, "w"), indent=1, ensure_ascii=False)
print(out, len(cells), "cells")
