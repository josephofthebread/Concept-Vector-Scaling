#!/bin/bash
#SBATCH --job-name=suspicious-vector-geometry
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/vector-geometry-%j.out
#SBATCH --error=logs/vector-geometry-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'EOF'
import numpy as np
from safetensors import safe_open

VEC = "../exp_genvectors/work/vectors/Qwen_Qwen3-8B--assistant--semantic"
for construction in ("diff", "whitened_diff"):
    with safe_open(f"{VEC}/{construction}.safetensors", framework="np") as handle:
        tensor = handle.get_tensor(construction)  # [layers, pairs, hidden]
    layers = tensor.shape[0]
    print(f"########## {construction}  {tensor.shape}")
    print(f"{'L':>3} {'cos(v,mean) mean':>17} {'median':>7} {'p10':>6} {'p90':>6} "
          f"{'|mean|/mean|v|':>15} {'eig1':>6} {'eig2':>6} {'eig3':>6} {'n50%':>5} {'n90%':>5} {'partR':>6}")
    for layer in range(layers):
        V = tensor[layer].astype(np.float64)
        V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)  # unit rows
        mean_vector = V.mean(axis=0)
        shared = np.linalg.norm(mean_vector)  # |mean of unit vectors|: 1 = identical, ~0 = isotropic
        cosines = V @ (mean_vector / max(shared, 1e-12))
        eig = np.linalg.eigvalsh(V.T @ V / V.shape[0])[::-1]
        share = eig / eig.sum()
        cumulative = np.cumsum(share)
        participation = float(eig.sum() ** 2 / (eig ** 2).sum())  # effective dimension
        print(f"{layer:>3} {cosines.mean():17.3f} {np.median(cosines):7.3f} "
              f"{np.percentile(cosines, 10):6.3f} {np.percentile(cosines, 90):6.3f} "
              f"{shared:15.3f} {share[0]:6.3f} {share[1]:6.3f} {share[2]:6.3f} "
              f"{int(np.searchsorted(cumulative, 0.5)) + 1:5d} "
              f"{int(np.searchsorted(cumulative, 0.9)) + 1:5d} {participation:6.1f}")
EOF
