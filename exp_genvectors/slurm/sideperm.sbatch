#!/bin/bash
#SBATCH --job-name=genvectors-sideperm
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/sideperm-%j.out
#SBATCH --error=logs/sideperm-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'EOF'
import numpy as np
from safetensors import safe_open
from exp_genvectors.artifacts import atomic_safetensors
from pathlib import Path

with safe_open("work/captures/Qwen_Qwen3-8B--assistant--semantic/shard-000.safetensors", framework="np") as f:
    counts = f.get_tensor("counts").astype(np.float32)
    sums = f.get_tensor("sums").astype(np.float32)  # packed: bucket means [L,P,2,B,H]
L, P, _, B, H = sums.shape
w = counts / counts.sum(axis=2, keepdims=True)
mu = np.einsum("lpsbh,psb->lpsh", sums, w)  # pole means [L,P,2,H]
shift = np.roll(np.arange(P), 1)  # deterministic derangement: j = i-1 mod P

def diag(v):
    unit = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    mean = unit.mean(axis=0)
    align = float((unit @ mean > 0).mean())
    return align, float(np.linalg.norm(mean))

variants = {
    "real (c_i - a_i)":            lambda l: mu[l, :, 0] - mu[l, :, 1],
    "perm sides kept (c_i - a_j)": lambda l: mu[l, :, 0] - mu[l, shift, 1],
    "same side concept (c_i - c_j)": lambda l: mu[l, :, 0] - mu[l, shift, 0],
    "same side antag (a_i - a_j)": lambda l: mu[l, :, 1] - mu[l, shift, 1],
}
out = {}
for name, fn in variants.items():
    aligns = np.empty(L, dtype=np.float32); norms = np.empty(L, dtype=np.float32)
    for l in range(L):
        aligns[l], norms[l] = diag(fn(l))
    out[name.replace(" ", "_") + "__align"] = aligns
    out[name.replace(" ", "_") + "__meannorm"] = norms
    picks = [0, 12, 19, 26, 33]
    print(f"{name:28s} align: " + " ".join(f"L{l}:{aligns[l]:.2f}" for l in picks)
          + "   |mean unit|: " + " ".join(f"{norms[l]:.3f}" for l in picks), flush=True)
print(f"chance |mean unit| = {1/np.sqrt(P):.3f}, chance align = 0.50")
atomic_safetensors(out, Path("work/convergence/side_permutations.safetensors"), {"note": "variant battery"})
EOF
