#!/bin/bash
#SBATCH --job-name=suspicious-cosine-hist
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/cosine-hist-%j.out
#SBATCH --error=logs/cosine-hist-%j.err
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

VECTORS = "../exp_gendata/work/vectors"
# ordered: 3 base tags, then 5 instruct tags
# named_{raw,assistant,assistant_decl} dropped: naming-drift bias makes them uninformative here.
TAGS = ["raw", "paired_raw", "user", "assistant", "paired_assistant"]
SHORT = {"raw": "raw", "paired_raw": "paired_raw",
         "user": "user", "assistant": "assistant", "paired_assistant": "paired_asst"}
FAMILIES = {
    "qwen":  ("Qwen_Qwen3-8B-Base", "Qwen_Qwen3-8B"),
    "gemma": ("google_gemma-2-9b", "google_gemma-2-9b-it"),
    "llama": ("unsloth_Meta-Llama-3.1-8B", "unsloth_Meta-Llama-3.1-8B-Instruct"),
}
BASE_TAGS = {"raw", "named_raw", "paired_raw"}
EPS = 1e-7
NBINS = 50


def tag_dir(base, inst, tag):
    stem = base if tag in BASE_TAGS else inst
    return f"{VECTORS}/{stem}--{tag}--semantic/diff.safetensors"


def load_unit(path):
    with safe_open(path, framework="np") as handle:
        matrix = handle.get_tensor("diff").astype(np.float32)   # [L, K, H]
    norm = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norm, 1e-9)


for family, (base, inst) in FAMILIES.items():
    present, loaded = [], {}
    for tag in TAGS:
        path = tag_dir(base, inst, tag)
        if os.path.exists(path):
            loaded[tag] = load_unit(path)
            present.append(tag)
        else:
            print(f"[{family}] MISSING {path}")
    if len(present) < 2:
        print(f"[{family}] too few tags, skipping")
        continue
    layers = loaded[present[0]].shape[0]
    pairs = [(a, b) for i, a in enumerate(present) for b in present[i + 1:]]

    # log(1 - cos) per pair, all layers at once: dist[(a,b)] -> [L, K]
    dist = {}
    for a, b in pairs:
        cos = np.einsum("lkh,lkh->lk", loaded[a], loaded[b])
        cos = np.clip(cos, -1.0, 1.0 - EPS)
        dist[(a, b)] = np.log(1.0 - cos).astype(np.float32)

    stacked = np.concatenate([d.ravel() for d in dist.values()])
    lo, hi = np.percentile(stacked, [0.05, 99.95])
    edges = np.linspace(lo, hi, NBINS + 1)

    n = len(present)
    outdir = f"work/figures/cosine-hist/{family}"
    os.makedirs(outdir, exist_ok=True)
    for layer in range(layers):
        fig, axes = plt.subplots(n, n, figsize=(2.0 * n, 2.0 * n), squeeze=False)
        for i in range(n):
            for j in range(n):
                ax = axes[i][j]
                ax.set_xticks([]); ax.set_yticks([])
                if i == j:
                    ax.text(0.5, 0.5, SHORT[present[i]], ha="center", va="center",
                            fontsize=9, fontweight="bold", transform=ax.transAxes)
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    continue
                if i > j:
                    ax.axis("off"); continue
                d = dist[(present[i], present[j])][layer]
                cross = (present[i] in BASE_TAGS) != (present[j] in BASE_TAGS)
                color = "#c44" if cross else "#369"
                ax.hist(d, bins=edges, color=color, alpha=0.85, linewidth=0)
                ax.axvline(np.median(d), color="k", linewidth=0.8)
                ax.set_xlim(lo, hi)
        fig.suptitle(f"{family}  layer {layer}   |   log(1 - cos) of diff vectors, per concept "
                     f"(red = base×instruct pair)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = f"{outdir}/layer-{layer:02d}.svg"
        fig.savefig(out)
        plt.close(fig)
    print(f"[{family}] wrote {layers} SVGs to {outdir}  ({n} tags, {len(pairs)} pairs/cell-matrix)")
EOF
