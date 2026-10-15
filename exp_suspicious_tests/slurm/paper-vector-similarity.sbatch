#!/bin/bash
#SBATCH --job-name=paper-vector-similarity
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/paper-vecsim-%j.out
#SBATCH --error=logs/paper-vecsim-%j.err
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
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib import gridspec

VECTORS = "../exp_genvectors/work/vectors"
TAGS = ["raw", "paired_raw", "user", "assistant", "paired_assistant"]
SHORT = ["raw", "paired", "user", "asst", "paired-asst"]
BASE_TAGS = {"raw", "paired_raw"}
EPS = 1e-7
QWEN = ("Qwen_Qwen3-8B-Base", "Qwen_Qwen3-8B")
LLAMA = ("unsloth_Meta-Llama-3.1-8B", "unsloth_Meta-Llama-3.1-8B-Instruct")
QWEN_LAYER = 24          # 2L/3 of 36
LLAMA_TOP, LLAMA_BOT = 21, 16   # 2L/3 and L/2 of 32
n = len(TAGS)


def load_unit(family, tag):
    stem = family[0] if tag in BASE_TAGS else family[1]
    with safe_open(f"{VECTORS}/{stem}--{tag}--semantic/diff.safetensors", framework="np") as h:
        m = h.get_tensor("diff").astype(np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9)


def pair_dists(family, layer):
    loaded = {t: load_unit(family, t)[layer] for t in TAGS}   # [K, H] each
    out = {}
    for i in range(n):
        for j in range(i + 1, n):
            cos = np.einsum("kh,kh->k", loaded[TAGS[i]], loaded[TAGS[j]])
            out[(i, j)] = np.log(1.0 - np.clip(cos, -1.0, 1.0 - EPS))
    return out


fig = plt.figure(figsize=(10.5, 3.4))
outer = gridspec.GridSpec(1, 2, width_ratios=[33, 67], wspace=0.14,
                          left=0.02, right=0.985, top=0.97, bottom=0.10)

# ---------------- LEFT: qwen tails heatmap, layer 24 ----------------
d_q = pair_dists(QWEN, QWEN_LAYER)
q95 = {k: float(np.percentile(v, 95)) for k, v in d_q.items()}
q05 = {k: float(np.percentile(v, 5)) for k, v in d_q.items()}
u_vals = np.array(list(q95.values())); l_vals = np.array(list(q05.values()))
u_mean, u_std = u_vals.mean(), u_vals.std() or 1.0
l_mean, l_std = l_vals.mean(), l_vals.std() or 1.0
zmax = max(np.abs((u_vals - u_mean) / u_std).max(), np.abs((l_vals - l_mean) / l_std).max())
norm_z = TwoSlopeNorm(vmin=-zmax, vcenter=0.0, vmax=zmax)
cmap_z = plt.get_cmap("RdBu_r")

axL = fig.add_subplot(outer[0])
Z = np.full((n, n), np.nan); RAW = np.full((n, n), np.nan)
for (i, j) in q95:
    Z[i, j] = (q95[(i, j)] - u_mean) / u_std; RAW[i, j] = q95[(i, j)]
    Z[j, i] = (q05[(i, j)] - l_mean) / l_std; RAW[j, i] = q05[(i, j)]
axL.imshow(np.ma.masked_invalid(Z), cmap=cmap_z, norm=norm_z, aspect="equal")
for i in range(n):
    for j in range(n):
        if i != j:
            axL.text(j, i, f"{RAW[i, j]:.2f}", ha="center", va="center", fontsize=8)
        else:
            axL.text(j, i, SHORT[i], ha="center", va="center", fontsize=8, rotation=45)
axL.set_xticks([]); axL.set_yticks([])       # diagonal carries the names
axL.set_xticks(np.arange(-.5, n, 1), minor=True)
axL.set_yticks(np.arange(-.5, n, 1), minor=True)
axL.grid(which="minor", color="white", linewidth=1.2)
axL.tick_params(which="both", length=0)
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_z, norm=norm_z), ax=axL,
                    fraction=0.045, pad=0.03)
cbar.ax.tick_params(labelsize=6.5)

# ---------------- RIGHT: one square matrix; upper = layer 21 (2L/3), lower = layer 16 (L/2) ----
d_top = pair_dists(LLAMA, LLAMA_TOP)
d_bot = pair_dists(LLAMA, LLAMA_BOT)
allv = np.concatenate([np.concatenate(list(d.values())) for d in (d_top, d_bot)])
lo, hi = np.percentile(allv, [0.1, 99.9])
edges = np.linspace(lo, hi, 36)
centers = 0.5 * (edges[:-1] + edges[1:])
norm_x = Normalize(vmin=lo, vmax=hi)
cmap_x = plt.get_cmap("coolwarm")

inner = outer[1].subgridspec(n, n, wspace=0.08, hspace=0.08)
for i in range(n):
    for j in range(n):
        ax = fig.add_subplot(inner[i, j])
        ax.set_xticks([]); ax.set_yticks([])
        if i == j:
            ax.text(0.5, 0.5, SHORT[i], ha="center", va="center", fontsize=7.5,
                    transform=ax.transAxes)
            for s in ax.spines.values():
                s.set_visible(False)
            continue
        # upper triangle (i<j): layer 2L/3; lower triangle (i>j): layer L/2 (mirrored pair)
        d = d_top[(i, j)] if i < j else d_bot[(j, i)]
        counts, _ = np.histogram(np.clip(d, lo, hi), bins=edges)
        ax.bar(centers, counts, width=edges[1] - edges[0],
               color=cmap_x(norm_x(centers)), linewidth=0)
        top = counts.max() or 1
        ax.set_xlim(lo, hi); ax.set_ylim(0, top * 1.25)
        for q, c in ((np.percentile(d, 5), cmap_x(0.0)), (np.percentile(d, 95), cmap_x(1.0))):
            ax.plot([q], [top * 0.06], marker="^", color=c, markersize=2.6, clip_on=False)
        for s in ax.spines.values():
            s.set_linewidth(0.4)
        if i == n - 1:                       # x labels on the bottom row only
            ticks = [t for t in (-8, -6, -4, -2) if lo <= t <= hi]
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks], fontsize=6)
            ax.tick_params(axis="x", length=1.5, pad=1)
for out in ("work/figures/vector-similarity.svg", "work/figures/vector-similarity.pdf"):
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
EOF
