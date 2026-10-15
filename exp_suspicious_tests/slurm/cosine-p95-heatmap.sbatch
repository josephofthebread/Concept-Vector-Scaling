#!/bin/bash
#SBATCH --job-name=suspicious-p95-heatmap
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/p95-heatmap-%j.out
#SBATCH --error=logs/p95-heatmap-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs work/figures
srun .venv/bin/python - <<'EOF'
import os
import numpy as np
from safetensors import safe_open
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

VECTORS = "../exp_genvectors/work/vectors"
BASE, INST = "Qwen_Qwen3-8B-Base", "Qwen_Qwen3-8B"
# named_{raw,assistant,assistant_decl} dropped: naming-drift bias makes them uninformative here.
TAGS = ["raw", "paired_raw", "user", "assistant", "paired_assistant"]
SHORT = ["raw", "paired_raw", "user", "assistant", "paired_asst"]
BASE_TAGS = {"raw", "named_raw", "paired_raw"}
LAYERS = [18, 19, 20, 21, 22, 23]
EPS = 1e-7


def load_unit(tag):
    stem = BASE if tag in BASE_TAGS else INST
    with safe_open(f"{VECTORS}/{stem}--{tag}--semantic/diff.safetensors", framework="np") as h:
        m = h.get_tensor("diff").astype(np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9)


loaded = {t: load_unit(t) for t in TAGS}
n = len(TAGS)

# p95 of log(1-cos) per (layer, i<j)
p95 = {}
for i in range(n):
    for j in range(i + 1, n):
        cos = np.einsum("lkh,lkh->lk", loaded[TAGS[i]], loaded[TAGS[j]])
        d = np.log(1.0 - np.clip(cos, -1.0, 1.0 - EPS))
        p95[(i, j)] = np.percentile(d, 95, axis=1)          # [L]

allvals = np.concatenate([p95[k][LAYERS] for k in p95])
vmin, vmax = allvals.min(), allvals.max()
vcenter = float(np.median(allvals))
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("#e8e8e8")

fig, axes = plt.subplots(2, 3, figsize=(22, 15))
for ax, layer in zip(axes.flat, LAYERS):
    M = np.full((n, n), np.nan)
    for (i, j), series in p95.items():
        M[i, j] = series[layer]
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, aspect="equal")
    for i in range(n):
        for j in range(i + 1, n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")
    ax.set_xticks(range(n)); ax.set_xticklabels(SHORT, rotation=45, ha="left", fontsize=8)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(n)); ax.set_yticklabels(SHORT, fontsize=8)
    ax.set_title(f"layer {layer}", fontsize=11, pad=28)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label("p95 of log(1 - cos)  (higher = heavier worst-case disagreement)", fontsize=10)
fig.suptitle("qwen  |  p95 of log(1-cos) between tag diff-vectors, per concept  |  layers 18-23",
             fontsize=13)
out = "work/figures/qwen-p95-L18-23.svg"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
EOF
