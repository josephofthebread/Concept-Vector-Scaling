#!/bin/bash
#SBATCH --job-name=basis-k
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/basis-k-%j.out
#SBATCH --error=logs/basis-k-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'EOF'
import numpy as np
from safetensors import safe_open

LAYER = 25
with safe_open("../exp_gendata/work/vectors/Qwen_Qwen3-8B--assistant--semantic/diff.safetensors",
               framework="np") as h:
    M = h.get_tensor("diff")[LAYER].astype(np.float64)          # [1036, H]
M /= np.linalg.norm(M, axis=1, keepdims=True)

for name, X in (("uncentered", M), ("centered", M - M.mean(axis=0, keepdims=True))):
    s = np.linalg.svd(X, compute_uv=False)
    p = s ** 2
    pr = p.sum() ** 2 / (p ** 2).sum()
    frac = np.cumsum(p) / p.sum()
    k90 = int(np.searchsorted(frac, 0.90) + 1)
    k50 = int(np.searchsorted(frac, 0.50) + 1)
    print(f"{name:10s}  spectrum PR = {pr:6.2f}   K(50% energy) = {k50}   K(90% energy) = {k90}")
    print(f"{'':10s}  top-15 sigma^2 share: " + " ".join(f"{v:.3f}" for v in (p / p.sum())[:15]))
EOF
