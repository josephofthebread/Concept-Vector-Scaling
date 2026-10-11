#!/bin/bash
#SBATCH --job-name=suspicious-cosines
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/cosines-%j.out
#SBATCH --error=logs/cosines-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs work/figures
srun .venv/bin/python - <<'EOF'
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open

VECTORS = "../exp_gendata/work/vectors"
FAMILIES = {
    "qwen3-8b": ["Qwen_Qwen3-8B-Base--raw", "Qwen_Qwen3-8B-Base--named_raw",
                 "Qwen_Qwen3-8B--user", "Qwen_Qwen3-8B--assistant", "Qwen_Qwen3-8B--named_assistant"],
    "gemma-2-9b": ["google_gemma-2-9b--raw", "google_gemma-2-9b--named_raw",
                   "google_gemma-2-9b-it--user", "google_gemma-2-9b-it--assistant",
                   "google_gemma-2-9b-it--named_assistant"],
    "llama-3.1-8b": ["unsloth_Meta-Llama-3.1-8B--raw", "unsloth_Meta-Llama-3.1-8B--named_raw",
                     "unsloth_Meta-Llama-3.1-8B-Instruct--user",
                     "unsloth_Meta-Llama-3.1-8B-Instruct--assistant",
                     "unsloth_Meta-Llama-3.1-8B-Instruct--named_assistant"],
}
BINS = np.linspace(0.0, 1.0, 121)


def load_mid(tag):
    """Unit-normalized diff vectors for the middle third of the depth: [layers, pairs, hidden]."""
    with safe_open(f"{VECTORS}/{tag}--semantic/diff.safetensors", framework="np") as handle:
        shape = handle.get_slice("diff").get_shape()
        lo, hi = shape[0] // 3, 2 * shape[0] // 3
        matrix = handle.get_slice("diff")[lo:hi].astype(np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=-1, keepdims=True), 1e-9)


def short(tag):
    model, presentation = tag.split("--")
    instruct = presentation in ("user", "assistant", "named_assistant")
    return f"{'inst' if instruct else 'base'}/{presentation}"


for family, tags in FAMILIES.items():
    loaded = {tag: load_mid(tag) for tag in tags}
    layers = next(iter(loaded.values())).shape[0]
    size = len(tags)
    figure, axes = plt.subplots(size, size, figsize=(3.0 * size, 2.4 * size), sharex=True)
    for row, tag_a in enumerate(tags):
        for column, tag_b in enumerate(tags):
            axis = axes[row, column]
            if row == column:
                axis.text(0.5, 0.5, short(tag_a), ha="center", va="center", fontsize=11, weight="bold")
                axis.set_axis_off()
                continue
            cosines = np.einsum("lkh,lkh->lk", loaded[tag_a], loaded[tag_b]).ravel()
            axis.hist(cosines, bins=BINS, color="#2b6cb0")
            axis.axvline(float(np.median(cosines)), color="#c53030", linewidth=1)
            axis.set_title(f"median {np.median(cosines):.3f}  min {cosines.min():.2f}", fontsize=8)
            axis.set_yticks([])
            if row == size - 1:
                axis.set_xlabel("cosine", fontsize=8)
    figure.suptitle(f"{family}: cosine between diff vectors of different presentation tags "
                    f"(middle-third layers, {layers} layers x 1036 concepts per panel)", fontsize=12)
    figure.tight_layout()
    path = f"work/figures/cosines-{family}.svg"
    figure.savefig(path)
    plt.close(figure)
    print(f"wrote {path}")
    for tag_a in tags:
        row = " ".join(f"{np.median(np.einsum('lkh,lkh->lk', loaded[tag_a], loaded[tag_b])):.3f}"
                       for tag_b in tags)
        print(f"  {short(tag_a):22s} {row}")
    del loaded
EOF
