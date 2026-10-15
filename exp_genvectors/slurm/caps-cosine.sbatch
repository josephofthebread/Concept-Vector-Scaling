#!/bin/bash
#SBATCH --job-name=genvectors-capscos
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/capscos-%j.out
#SBATCH --error=logs/capscos-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'EOF'
import json
import numpy as np
from pathlib import Path
from safetensors import safe_open
from exp_genvectors.artifacts import atomic_safetensors

# concept-derived axes, exactly as the scope readout builds them
with safe_open("work/captures/Qwen_Qwen3-8B--assistant--semantic/shard-000.safetensors", framework="np") as f:
    counts = f.get_tensor("counts").astype(np.float32)
    unit_mu = f.get_tensor("unit_diff_sum").astype(np.float32)
    lang_mu = f.get_tensor("language_sum").astype(np.float32)
    lang_count = f.get_tensor("language_count").astype(np.float32)[:, 0]
with safe_open("work/sentiment/axis.safetensors", framework="np") as f:
    sentiment = f.get_tensor("axis").astype(np.float32)

def unitize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)

shared = unit_mu.mean(axis=2).mean(axis=1)
L = shared.shape[0]
v_lang = lang_mu[:, :, 0] - lang_mu[:, :, 1]
others = [i for i in range(5) if i != 2]
w_o = lang_count.mean(0)[others] / lang_count.mean(0)[others].sum()
offset = v_lang[:, :, 2] - np.einsum("lpoh,o->lph", v_lang[:, :, others], w_o)
mandarin = np.stack([np.linalg.svd(offset[l], full_matrices=False)[2][0] for l in range(L)])
shared_u, sent_u, mand_u = unitize(shared), unitize(sentiment), unitize(mandarin)
residual = shared_u - (shared_u * sent_u).sum(-1, keepdims=True) * sent_u
residual_u = unitize(residual - (residual * mand_u).sum(-1, keepdims=True) * mand_u)

# the six mechanical transform vectors (syntax run), pair order as in prepare.TRANSFORMS
with safe_open("work/vectors/Qwen_Qwen3-8B--assistant--syntax/diff.safetensors", framework="np") as f:
    syn = f.get_tensor("diff").astype(np.float32)                       # [L, 6, H]
TRANSFORMS = ["caps", "lowercase", "no_punctuation", "hard_newlines_50", "markdown_bullets", "identity_space"]
syn_u = unitize(syn)
AXES = {"control_shared": shared_u, "shared_residual": residual_u, "sentiment": sent_u, "mandarin": mand_u}
out = {}
for name, axis in AXES.items():
    cos = np.einsum("lh,lth->lt", axis, syn_u).astype(np.float32)       # [L, 6]
    out[name] = cos
    print(f"--- {name} ---", flush=True)
    for l in (0, 6, 12, 18, 25, 30, 36):
        print("  L%-3d " % l + "  ".join(f"{TRANSFORMS[t][:9]}={cos[l, t]:+.3f}" for t in range(6)), flush=True)
out["transform_norms"] = np.linalg.norm(syn, axis=2).astype(np.float32)
atomic_safetensors(out, Path("work/convergence/caps_cosine.safetensors"),
                   {"transforms": json.dumps(TRANSFORMS)})
print("done", flush=True)
EOF
