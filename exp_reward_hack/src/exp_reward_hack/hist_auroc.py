import glob
import json
import os
from collections import defaultdict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "."
MODELS = {"qwen": 23, "llama": 15, "gemma": 24}
KS = [1, 3, 5, 7]
THR = 0.85
MING = 10
labels = {r["file"]: r for r in json.load(open(f"{D}/labels.json"))}


fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
colors = {1: "#1f77b4", 3: "#2ca02c", 5: "#ff7f0e", 7: "#d62728"}
summary = {}
for ax, (mtag, L) in zip(axes, MODELS.items()):
    bytask = defaultdict(list)
    for f in sorted(glob.glob(f"{D}/turnwork_tok/*-{mtag}-A-seed*.npz")):
        lab = labels.get(os.path.basename(f).replace(".npz", ".json"))
        if lab is None:
            continue
        z = np.load(f, allow_pickle=True)
        bytask[lab["workload"]].append((bool(lab["cheated"]), z["z"].astype(np.float32), z["turn"]))
    summary[mtag] = {}
    for k in KS:
        vals = []
        for task, eps in bytask.items():
            ch = [(z, t) for c, z, t in eps if c]
            cl = [(z, t) for c, z, t in eps if not c]
            if len(ch) < MING or len(cl) < MING:
                continue
            pos = [z[t < k] for z, t in ch if (t < k).any()]
            neg = [z[t < k] for z, t in cl if (t < k).any()]
            if not pos or not neg:
                continue
            pos, neg = np.vstack(pos), np.vstack(neg)
            if len(pos) < 20 or len(neg) < 20:
                continue
            na, nb = len(pos), len(neg)
            X = np.vstack([pos, neg])
            ranks = np.argsort(np.argsort(X, axis=0), axis=0).astype(np.float64) + 1
            a = (ranks[:na].sum(0) - na * (na + 1) / 2.0) / (na * nb)
            vals.append(np.maximum(a, 1 - a))
        if not vals:
            continue
        allv = np.concatenate(vals)
        summary[mtag][k] = {
            "n": int(allv.size),
            "median": round(float(np.median(allv)), 3),
            "p99": round(float(np.percentile(allv, 99)), 3),
            "max": round(float(allv.max()), 3),
            "n_ge_thr": int((allv >= THR).sum()),
        }
        ax.hist(
            allv,
            bins=np.linspace(0.5, 1.0, 51),
            density=True,
            histtype="step",
            lw=1.8,
            color=colors[k],
            label=f"K≤{k} (max {allv.max():.2f})",
        )
    ax.axvline(THR, color="k", ls="--", lw=1.2)
    ax.text(THR + 0.005, ax.get_ylim()[1] * 0.9, "0.85", fontsize=9)
    ax.set_title(f"{mtag} (L{L})", fontsize=11)
    ax.set_xlabel("per-concept token-AUROC (cheat vs clean, within task)", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
axes[0].set_ylabel("density")
fig.suptitle(
    "Per-concept CoT-token AUROC distribution by model (within-task, K sweep) — "
    "mass sits near chance; nothing reaches 0.85",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs(f"{D}/xresults", exist_ok=True)
fig.savefig(f"{D}/xresults/tokauroc_hist.png", dpi=140)
json.dump(summary, open(f"{D}/xresults/tokauroc_hist.json", "w"), indent=2)
print("wrote xresults/tokauroc_hist.png")
for m in summary:
    for k in KS:
        if k in summary[m]:
            s = summary[m][k]
            print(
                f"  {m:6} K<={k}: n={s['n']:5} median={s['median']} p99={s['p99']} "
                f"max={s['max']} concepts>={THR}: {s['n_ge_thr']}"
            )
