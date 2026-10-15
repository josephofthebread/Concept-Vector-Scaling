#!/bin/bash
#SBATCH --job-name=genvectors-dims
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/dims-%j.out
#SBATCH --error=logs/dims-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'EOF'
import numpy as np
from pathlib import Path
from safetensors import safe_open
from exp_genvectors.artifacts import atomic_safetensors

RUNS = {"qwen": "Qwen_Qwen3-8B--assistant--semantic",
        "llama": "unsloth_Meta-Llama-3.1-8B-Instruct--assistant--semantic",
        "gemma": "google_gemma-2-9b-it--assistant--semantic"}

def stats(v):
    unit = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    x = unit - unit.mean(axis=0)
    s = np.linalg.svd(x, compute_uv=False)
    lam = s ** 2
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    cum = np.cumsum(lam) / lam.sum()
    return pr, int(np.searchsorted(cum, 0.5) + 1), int(np.searchsorted(cum, 0.9) + 1)

out = {}
for model, tag in RUNS.items():
    with safe_open(f"work/captures/{tag}/shard-000.safetensors", framework="np") as f:
        counts = f.get_tensor("counts").astype(np.float32)
        sums = f.get_tensor("sums").astype(np.float32)
    L, P = sums.shape[0], sums.shape[1]
    w = counts / counts.sum(axis=2, keepdims=True)
    mu = np.einsum("lpsbh,psb->lpsh", sums, w)
    shift = np.roll(np.arange(P), 1)
    rows = []
    for l in range(L):
        real = stats(mu[l, :, 0] - mu[l, :, 1])
        null = stats(mu[l, :, 0] - mu[l, shift, 1])   # sides-kept permuted-label null
        rows.append((*real, *null))
    arr = np.array(rows, dtype=np.float32)
    out[f"{model}"] = arr  # [L, 6]: PR, n50, n90, PR_null, n50_null, n90_null
    picks = [0, L//4, L//2, int(L*0.7), L-1]
    print(f"{model} (P={P}, {L} layers): layer: PR n50 n90 | null: PR n50 n90")
    for l in picks:
        r = arr[l]
        print(f"  L{l:2d}: {r[0]:6.1f} {int(r[1]):4d} {int(r[2]):4d}  |  {r[3]:6.1f} {int(r[4]):4d} {int(r[5]):4d}", flush=True)
atomic_safetensors(out, Path("work/convergence/dimensionality.safetensors"), {"axes": "layer x (PR,n50,n90,PRn,n50n,n90n)"})
EOF
