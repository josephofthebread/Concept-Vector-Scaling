#!/bin/bash
#SBATCH --job-name=genvectors-mandaxis
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/mandaxis-%j.out
#SBATCH --error=logs/mandaxis-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'EOF'
import json
import numpy as np
from pathlib import Path
from safetensors import safe_open
from exp_genvectors.artifacts import atomic_safetensors

MAND = 2
RUNS = {"qwen": "Qwen_Qwen3-8B--assistant--semantic"}
out = {}
for model, tag in RUNS.items():
    with safe_open(f"work/captures/{tag}/shard-000.safetensors", framework="np") as f:
        lang_mu = f.get_tensor("language_sum").astype(np.float32)   # packed: means [L,P,2,5,H]
        lang_count = f.get_tensor("language_count").astype(np.float32)[:, 0]  # [P,5]
        sums = f.get_tensor("sums").astype(np.float32)              # packed: bucket means [L,P,2,B,H]
        counts = f.get_tensor("counts").astype(np.float32)
        unit_mu = f.get_tensor("unit_diff_sum").astype(np.float32)  # packed means [L,P,B,H]
    L, P, _, _, H = lang_mu.shape
    v_lang = lang_mu[:, :, 0] - lang_mu[:, :, 1]                    # [L,P,5,H]
    weights = lang_count.mean(0) / lang_count.mean(0).sum()         # ~[0.2]*5
    others = [i for i in range(5) if i != MAND]
    w_o = lang_count.mean(0)[others] / lang_count.mean(0)[others].sum()
    v_rest = np.einsum("lpoh,o->lph", v_lang[:, :, others], w_o)
    offset = v_lang[:, :, MAND] - v_rest                            # [L,P,H]
    pooled = np.einsum("lpsh,s->lph", np.stack([v_lang[:, :, i] for i in range(5)], axis=2), weights) if False else np.einsum("lpih,i->lph", v_lang, weights)
    share = weights[MAND] * np.linalg.norm(offset, axis=2) / np.maximum(np.linalg.norm(pooled, axis=2), 1e-9)  # [L,P]
    pc1 = np.empty(L, dtype=np.float32); meancos = np.empty(L, dtype=np.float32)
    axis_vs_shared = np.empty(L, dtype=np.float32)
    for l in range(L):
        U = offset[l] / np.maximum(np.linalg.norm(offset[l], axis=1, keepdims=True), 1e-9)
        G = U @ U.T
        eig = np.linalg.eigvalsh(G)
        pc1[l] = eig[-1] / P
        meancos[l] = (np.abs(G)[np.triu_indices(P, 1)]).mean()
        # global mandarin axis = top left-singular vector of offsets
        _, s, vt = np.linalg.svd(offset[l], full_matrices=False)
        mand_axis = vt[0]
        # shared axis = mean of unit diff vectors (rows-weighted mean of bucket unit means)
        unit_mean = (unit_mu[l] * counts[:, 0][None if False else slice(None), :, None]).sum(axis=1)  # [P,H]*? simpler:
        unit_mean = unit_mu[l].mean(axis=1)  # [P,H] mean over buckets of unit-mean
        shared = unit_mean.mean(axis=0)
        axis_vs_shared[l] = abs(float(mand_axis @ shared / (np.linalg.norm(shared) + 1e-12)))
    out[f"{model}_share"] = share.astype(np.float32)
    out[f"{model}_pc1"] = pc1; out[f"{model}_meancos"] = meancos; out[f"{model}_axis_vs_shared"] = axis_vs_shared
    print(model, "done: median share@deep", float(np.median(share[int(L*0.7)])),
          "pc1@deep", float(pc1[int(L*0.7)]), "|cos(mand_axis, shared)|@deep", float(axis_vs_shared[int(L*0.7)]), flush=True)
atomic_safetensors(out, Path("work/convergence/mandarin_axis.safetensors"), {"note": "see langscores thread"})
EOF
