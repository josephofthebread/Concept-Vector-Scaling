#!/bin/bash
#SBATCH --job-name=gendata-f32check
#SBATCH --partition=test
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/f32check-%j.out
#SBATCH --error=logs/f32check-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'EOF'
import numpy as np
from safetensors import safe_open

from exp_gendata.vecmath import cosine, means, whiten, within_covariance

path = "work/captures/Qwen_Qwen3-8B-Base--raw--semantic/shard-000.safetensors"
with safe_open(path, framework="np") as handle:
    counts = handle.get_tensor("counts")
    sums_slice = handle.get_slice("sums")
    moments_slice = handle.get_slice("moments")
    for layer in (0, 9, 18, 27, 36):
        sums = np.asarray(sums_slice[layer : layer + 1])
        moments = np.asarray(moments_slice[layer])
        mu = means(sums, counts)[0]
        diff = mu[:, 0] - mu[:, 1]
        group_sums = sums[0].sum(axis=2).reshape(-1, sums.shape[-1])
        group_counts = counts.sum(axis=2).reshape(-1)
        cov64 = within_covariance(moments.sum(axis=0), group_sums, group_counts)
        cov32 = within_covariance(
            moments.astype(np.float32).astype(np.float64).sum(axis=0), group_sums, group_counts
        )
        white64 = whiten(diff, cov64, 0.05)
        white32 = whiten(diff, cov32, 0.05)
        agreement = cosine(white64, white32)
        eig64 = np.linalg.eigvalsh(cov64)
        eig32 = np.linalg.eigvalsh(cov32)
        relerr = np.abs(eig64 - eig32) / np.maximum(np.abs(eig64), 1e-30)
        print(
            f"L{layer:02d}: cos(white64, white32) min={agreement.min():.9f} median={np.median(agreement):.9f}  "
            f"eig relerr max={relerr.max():.2e}  smallest eig64={eig64.min():.3e}",
            flush=True,
        )
EOF
