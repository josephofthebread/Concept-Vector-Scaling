#!/bin/bash
#SBATCH --job-name=suspicious-cosine-ks
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/cosine-ks-%j.out
#SBATCH --error=logs/cosine-ks-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs work/figures
srun .venv/bin/python - <<'EOF'
import itertools
import json

import numpy as np
from safetensors import safe_open
from scipy.stats import ks_2samp

VECTORS = "../exp_gendata/work/vectors"
PRESENTATIONS = ["raw", "named_raw", "user", "assistant", "named_assistant"]
FAMILIES = {
    "qwen": {"raw": "Qwen_Qwen3-8B-Base--raw", "named_raw": "Qwen_Qwen3-8B-Base--named_raw",
             "user": "Qwen_Qwen3-8B--user", "assistant": "Qwen_Qwen3-8B--assistant",
             "named_assistant": "Qwen_Qwen3-8B--named_assistant"},
    "gemma": {"raw": "google_gemma-2-9b--raw", "named_raw": "google_gemma-2-9b--named_raw",
              "user": "google_gemma-2-9b-it--user", "assistant": "google_gemma-2-9b-it--assistant",
              "named_assistant": "google_gemma-2-9b-it--named_assistant"},
    "llama": {"raw": "unsloth_Meta-Llama-3.1-8B--raw", "named_raw": "unsloth_Meta-Llama-3.1-8B--named_raw",
              "user": "unsloth_Meta-Llama-3.1-8B-Instruct--user",
              "assistant": "unsloth_Meta-Llama-3.1-8B-Instruct--assistant",
              "named_assistant": "unsloth_Meta-Llama-3.1-8B-Instruct--named_assistant"},
}


def load_mid(tag):
    with safe_open(f"{VECTORS}/{tag}--semantic/diff.safetensors", framework="np") as handle:
        shape = handle.get_slice("diff").get_shape()
        lo, hi = shape[0] // 3, 2 * shape[0] // 3
        matrix = handle.get_slice("diff")[lo:hi].astype(np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=-1, keepdims=True), 1e-9)


cells = list(itertools.combinations(PRESENTATIONS, 2))
distributions = {}
for family, tags in FAMILIES.items():
    loaded = {presentation: load_mid(tag) for presentation, tag in tags.items()}
    for a, b in cells:
        distributions[(family, a, b)] = np.einsum("lkh,lkh->lk", loaded[a], loaded[b]).ravel()
    layers = next(iter(loaded.values())).shape[0]
    print(f"{family}: {layers} mid layers, {distributions[(family, *cells[0])].size} values per cell")
    del loaded

summary = {}
for family_a, family_b in itertools.combinations(FAMILIES, 2):
    print(f"\n=== KS statistic: {family_a} vs {family_b} (same cell, two-sample) ===")
    print(f"{'cell':34s} {'KS':>6s} {'median A':>9s} {'median B':>9s}")
    values = []
    for a, b in cells:
        left = distributions[(family_a, a, b)]
        right = distributions[(family_b, a, b)]
        statistic = float(ks_2samp(left, right).statistic)
        values.append(statistic)
        print(f"{a + ' vs ' + b:34s} {statistic:6.3f} {np.median(left):9.3f} {np.median(right):9.3f}")
    summary[f"{family_a}|{family_b}"] = dict(zip([f"{a}|{b}" for a, b in cells], values))
    print(f"{'mean':34s} {np.mean(values):6.3f}")

with open("work/figures/cosine-ks.json", "w") as stream:
    json.dump(summary, stream, indent=2)
print("\nwrote work/figures/cosine-ks.json")
EOF
