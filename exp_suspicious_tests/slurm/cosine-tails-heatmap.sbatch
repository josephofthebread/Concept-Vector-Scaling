#!/bin/bash
#SBATCH --job-name=suspicious-tails-heatmap
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/tails-heatmap-%j.out
#SBATCH --error=logs/tails-heatmap-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs work/figures
srun .venv/bin/python - <<'EOF'
import numpy as np
from safetensors import safe_open
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

VECTORS = "../exp_genvectors/work/vectors"
BASE, INST = "Qwen_Qwen3-8B-Base", "Qwen_Qwen3-8B"
TAGS = ["raw", "paired_raw", "user", "assistant", "paired_assistant"]
SHORT = ["raw", "paired_raw", "user", "assistant", "paired_asst"]
BASE_TAGS = {"raw", "paired_raw"}
LAYERS = [18, 19, 20, 21, 22, 23]
EPS = 1e-7


def load_unit(tag):
    stem = BASE if tag in BASE_TAGS else INST
    with safe_open(f"{VECTORS}/{stem}--{tag}--semantic/diff.safetensors", framework="np") as h:
        m = h.get_tensor("diff").astype(np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9)


loaded = {t: load_unit(t) for t in TAGS}
n = len(TAGS)

q95, q05 = {}, {}
for i in range(n):
    for j in range(i + 1, n):
        cos = np.einsum("lkh,lkh->lk", loaded[TAGS[i]], loaded[TAGS[j]])
        d = np.log(1.0 - np.clip(cos, -1.0, 1.0 - EPS))
        q95[(i, j)] = np.percentile(d, 95, axis=1)   # [L]  right tail: worst disagreement
        q05[(i, j)] = np.percentile(d, 5, axis=1)    # [L]  left  tail: best agreement


# Standardize each tail to a within-tail z-score so ONE colorbar covers both triangles.
# In log(1-cos), lower raw value = more aligned, so for both tails z<0 = better, z>0 = worse.
up_all = np.concatenate([q95[k][LAYERS] for k in q95])
lo_all = np.concatenate([q05[k][LAYERS] for k in q05])
u_mean, u_std = float(up_all.mean()), float(up_all.std()) or 1.0
l_mean, l_std = float(lo_all.mean()), float(lo_all.std()) or 1.0
zmax = max(np.abs((up_all - u_mean) / u_std).max(), np.abs((lo_all - l_mean) / l_std).max())
norm = TwoSlopeNorm(vmin=-zmax, vcenter=0.0, vmax=zmax)
cmap = plt.get_cmap("RdBu_r").copy()

fig, axes = plt.subplots(2, 3, figsize=(22, 15))
for ax, layer in zip(axes.flat, LAYERS):
    Z = np.full((n, n), np.nan)                      # color = within-tail z-score
    RAW = np.full((n, n), np.nan)                    # annotation = raw q5/q95
    for (i, j) in q95:
        Z[i, j] = (q95[(i, j)][layer] - u_mean) / u_std   # upper
        RAW[i, j] = q95[(i, j)][layer]
        Z[j, i] = (q05[(i, j)][layer] - l_mean) / l_std   # lower (mirror)
        RAW[j, i] = q05[(i, j)][layer]
    ax.imshow(np.ma.masked_invalid(Z), cmap=cmap, norm=norm, aspect="equal")
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{RAW[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")
            else:
                ax.text(j, i, SHORT[i], ha="center", va="center", fontsize=8, fontweight="bold")
    ax.set_xticks(range(n)); ax.set_xticklabels(SHORT, rotation=45, ha="left", fontsize=8)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(n)); ax.set_yticklabels(SHORT, fontsize=8)
    ax.set_title(f"layer {layer}", fontsize=11, pad=30)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes,
                    fraction=0.022, pad=0.02)
cbar.set_label("within-tail z-score   (blue < 0 = more aligned,  red > 0 = less aligned)", fontsize=10)
fig.suptitle("qwen  |  log(1-cos) tails between tag diff-vectors  |  "
             "UPPER=q95 (worst disagreement)   LOWER=q5 (best agreement)   |  "
             "cells show raw values; lower = more aligned", fontsize=12)
out = "work/figures/qwen-tails-L18-23.svg"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
EOF
