#!/bin/bash
#SBATCH --job-name=paper-pr-depth
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/paper-pr-%j.out
#SBATCH --error=logs/paper-pr-%j.err
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

VECTORS = "../exp_genvectors/work/vectors"
TAGS = ["raw", "paired_raw", "user", "assistant", "paired_assistant"]
LABEL = {"raw": "raw", "paired_raw": "paired", "user": "user",
         "assistant": "assistant", "paired_assistant": "paired-assistant"}
COLOR = {"raw": "#1f77b4", "paired_raw": "#17becf", "user": "#ff7f0e",
         "assistant": "#d62728", "paired_assistant": "#9467bd"}
STYLE = {"raw": "--", "paired_raw": "--", "user": "-", "assistant": "-", "paired_assistant": "-"}
BASE_TAGS = {"raw", "paired_raw"}
FAMILIES = [("Qwen3-8B", "Qwen_Qwen3-8B-Base", "Qwen_Qwen3-8B"),
            ("gemma-2-9b", "google_gemma-2-9b", "google_gemma-2-9b-it"),
            ("Llama-3.1-8B", "unsloth_Meta-Llama-3.1-8B", "unsloth_Meta-Llama-3.1-8B-Instruct")]

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharey=False)
for ax, (name, base, inst) in zip(axes, FAMILIES):
    for tag in TAGS:
        stem = base if tag in BASE_TAGS else inst
        with safe_open(f"{VECTORS}/{stem}--{tag}--semantic/diff.safetensors", framework="np") as h:
            v = h.get_tensor("diff").astype(np.float64)          # [L, K, H]
        s2 = np.sum(v * v, axis=-1)                              # [L, K]
        s4 = np.sum(v ** 4, axis=-1)
        pr = np.median(s2 * s2 / np.maximum(s4, 1e-30), axis=1)  # [L]
        layers = np.arange(pr.shape[0])
        ax.plot(layers, pr, STYLE[tag], color=COLOR[tag], linewidth=1.4, label=LABEL[tag])
        print(f"{name} {tag}: PR at 2L/3 = {pr[int(round(2 * (pr.shape[0] - 1) / 3))]:.0f}")
    ax.set_xlabel("layer", fontsize=8)
    ax.set_title(name, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, layers[-1])
axes[0].set_ylabel("median participation ratio", fontsize=8)
axes[0].legend(fontsize=6.5, frameon=False)
fig.tight_layout()
for out in ("work/figures/pr-depth.svg", "work/figures/pr-depth.pdf"):
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
EOF
