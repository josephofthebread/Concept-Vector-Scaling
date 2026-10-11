import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TAGS = ["qwen", "llama", "gemma"]
NICE = {"qwen": "Qwen3-8B", "llama": "Llama-3.1-8B", "gemma": "gemma-2-9b"}
LAD = {"tylenol": "#d62728", "syrup": "#ff7f0e", "ibuprofen": "#9467bd", "steps": "#1f77b4"}
DANGER = ["tylenol", "syrup", "ibuprofen"]
RANDOM_COL = 457
LI = 5

res = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "results/dose-response.pdf")
feats = json.load(open("../exp_lmsys_probe/data/features.json"))["features"]
feats.sort(key=lambda f: f["pair_id"])
antag = {f["pair_id"]: f["antagonist"] for f in feats}


fig, axes = plt.subplots(3, 3, figsize=(12, 9.6), sharey="row")
summary = []
for r, t in enumerate(TAGS):
    npz = np.load(res / f"dose_{t}.npz", allow_pickle=True)
    meta = json.loads(str(npz["meta"]))
    layer = meta["layers"][LI]
    names = meta["names"]

    def scores(key):
        danger = np.mean([npz[f"{key}.{l}"][-1, LI] - npz[f"{key}.{l}"][0, LI] for l in DANGER], axis=0)
        arr = npz[f"{key}.steps"]
        steps = arr[-1, LI] - arr[0, LI]
        return danger - steps, steps

    cc, cs = scores("cos")
    xc, xs = scores("ctl")
    z = (cc - xc.mean()) / xc.std()
    zs = (cs - xs.mean()) / xs.std()
    zr = float((xc[RANDOM_COL] - xc.mean()) / xc.std())
    top = np.argsort(-z)[:2]
    row = {
        "model": t,
        "layer": layer,
        "n_z_gt5": int((z > 5).sum()),
        "n_z_gt10": int((z > 10).sum()),
        "null_std_dcos": round(float(xc.std()), 5),
        "random_col_z": round(zr, 2),
        "top": [],
    }
    for cbox, pid in enumerate(top):
        ax = axes[r][cbox]
        for lad, color in LAD.items():
            a = npz[f"cos.{lad}"][:, LI, pid]
            ax.plot(meta["ladders"][lad]["doses"], a - a[0], "o-", color=color, ms=4, lw=1.6, label=lad)
        ax.set_title(f"#{pid} {names[pid]}\nz={z[pid]:.1f} steps={zs[pid]:.1f}", fontsize=9)
        row["top"].append(
            {
                "pid": int(pid),
                "concept": names[pid],
                "antagonist": antag[int(pid)],
                "z": round(float(z[pid]), 1),
                "steps_z": round(float(zs[pid]), 1),
                "dcos_tylenol": round(float((npz["cos.tylenol"][-1, LI] - npz["cos.tylenol"][0, LI])[pid]), 4),
                "dcos_ibuprofen": round(float((npz["cos.ibuprofen"][-1, LI] - npz["cos.ibuprofen"][0, LI])[pid]), 4),
                "dcos_steps": round(float((npz["cos.steps"][-1, LI] - npz["cos.steps"][0, LI])[pid]), 4),
            }
        )
    ax = axes[r][2]
    for lad, color in LAD.items():
        a = npz[f"ctl.{lad}"][:, LI, RANDOM_COL]
        ax.plot(meta["ladders"][lad]["doses"], a - a[0], "o-", color=color, ms=4, lw=1.6, label=lad)
    ax.set_title(f"random #{RANDOM_COL}\nz={zr:.1f}", fontsize=9)
    for ax in axes[r]:
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("dose")
    axes[r][0].set_ylabel(f"{NICE[t]} (layer {layer})\n$\\Delta$cos, assistant token")
    summary.append(row)

axes[0][0].legend(fontsize=9, loc="upper left")
fig.tight_layout()
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
fig.savefig(out.with_suffix(".png"), dpi=140)
print(json.dumps(summary, indent=1))
print("wrote", out)
