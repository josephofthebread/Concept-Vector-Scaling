#!/bin/bash
#SBATCH --job-name=gendata-sentana
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/sentana-%j.out
#SBATCH --error=logs/sentana-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'EOF'
import json
import numpy as np
from pathlib import Path
from safetensors import safe_open
from exp_gendata.artifacts import atomic_safetensors

with safe_open("work/sentiment/axis.safetensors", framework="np") as f:
    sent_axis = f.get_tensor("axis")            # [37, H]
with safe_open("work/sentiment/pooled.safetensors", framework="np") as f:
    pooled = f.get_tensor("pooled").astype(np.float32)  # [4000, 37, H]
    stars = f.get_tensor("stars")
with safe_open("work/captures/Qwen_Qwen3-8B--assistant--semantic/shard-000.safetensors", framework="np") as f:
    counts = f.get_tensor("counts").astype(np.float32)
    unit_mu = f.get_tensor("unit_diff_sum").astype(np.float32)   # packed means [L,P,B,H]
    lang_mu = f.get_tensor("language_sum").astype(np.float32)
    lang_count = f.get_tensor("language_count").astype(np.float32)[:, 0]
L, P = unit_mu.shape[0], unit_mu.shape[1]
shared = unit_mu.mean(axis=2).mean(axis=1)      # [L, H]
v_lang = lang_mu[:, :, 0] - lang_mu[:, :, 1]
others = [i for i in range(5) if i != 2]
w_o = lang_count.mean(0)[others] / lang_count.mean(0)[others].sum()
offset = v_lang[:, :, 2] - np.einsum("lpoh,o->lph", v_lang[:, :, others], w_o)
mand_axis = np.empty_like(shared)
for l in range(L):
    _, _, vt = np.linalg.svd(offset[l], full_matrices=False)
    mand_axis[l] = vt[0]

def cosv(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

cos_shared = np.array([cosv(sent_axis[l], shared[l]) for l in range(L)], dtype=np.float32)
cos_mand = np.array([abs(cosv(sent_axis[l], mand_axis[l])) for l in range(L)], dtype=np.float32)
proj = np.einsum("rlh,lh->rl", pooled, sent_axis / np.linalg.norm(sent_axis, axis=1, keepdims=True))
out = {"cos_shared": cos_shared, "cos_mandarin": cos_mand, "projections": proj.astype(np.float32),
       "stars": stars}
atomic_safetensors(out, Path("work/sentiment/analysis.safetensors"), {"note": "sentiment axis vs shared/mandarin"})
for l in (0, 12, 19, 26, 30, 33, 36):
    med = {s: float(np.median(proj[stars == s, l])) for s in (1, 2, 4, 5)}
    print(f"L{l:2d}: cos(sent,shared)={cos_shared[l]:+.3f}  cos(sent,mand)={cos_mand[l]:.3f}  "
          f"median proj 1/2/4/5 star: {med[1]:.1f} {med[2]:.1f} {med[4]:.1f} {med[5]:.1f}", flush=True)
EOF
