#!/bin/bash
#SBATCH --job-name=gendata-rnorm
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/rnorm-%j.out
#SBATCH --error=logs/rnorm-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'EOF'
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

TAGS = ["Qwen_Qwen3-8B-Base--raw--semantic", "Qwen_Qwen3-8B-Base--named_raw--semantic",
        "Qwen_Qwen3-8B--user--semantic", "Qwen_Qwen3-8B--assistant--semantic",
        "Qwen_Qwen3-8B--named_assistant--semantic"]
out = {}
for tag in TAGS:
    with safe_open(f"work/captures/{tag}/shard-000.safetensors", framework="np") as f:
        counts = f.get_tensor("counts").astype(np.float32)
        rows = counts[:, 0, :]
        unit = f.get_tensor("unit_diff_sum").astype(np.float32) * np.maximum(rows, 1)[None, :, :, None]
        R = np.linalg.norm(unit.sum(axis=2), axis=-1) / rows.sum(axis=1)[None, :]
        del unit
        sums = f.get_tensor("sums").astype(np.float32) * np.maximum(counts, 1)[None, ..., None]
        mu = sums.sum(axis=3) / counts.sum(axis=2)[None, ..., None]
        norm = np.linalg.norm(mu[:, :, 0] - mu[:, :, 1], axis=-1)
        del sums
    out[f"R__{tag}"] = R.astype(np.float32)
    out[f"norm__{tag}"] = norm.astype(np.float32)
    print("done", tag, flush=True)
save_file(out, "work/convergence/qwen_R_norm.safetensors")
EOF
