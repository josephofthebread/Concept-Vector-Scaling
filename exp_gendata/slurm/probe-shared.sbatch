#!/bin/bash
#SBATCH --job-name=gendata-probeshared
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=logs/probeshared-%j.out
#SBATCH --error=logs/probeshared-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'EOF'
import numpy as np
from pathlib import Path
from scipy.stats import norm
from safetensors import safe_open
from exp_gendata.artifacts import atomic_safetensors

CAP = "work/captures/Qwen_Qwen3-8B--assistant--semantic/shard-000.safetensors"
with safe_open(CAP, framework="np") as f:
    counts = f.get_tensor("counts").astype(np.float64)          # [P,2,B]
    sums = f.get_tensor("sums").astype(np.float32)              # packed bucket MEANS [L,P,2,B,H]
    unit_mu = f.get_tensor("unit_diff_sum").astype(np.float32)  # packed means [L,P,B,H]
    L, P, _, B, H = sums.shape
    tri_r, tri_c = np.triu_indices(H)
    mom = f.get_slice("moments_tri")

TRAIN, TEST = [0, 1], [2, 3]
auroc = np.zeros((2, L, P), dtype=np.float32)                   # [variant, layer, pair]

def pole_mean(layer, buckets):
    w = counts[:, :, buckets]
    return (sums[layer][:, :, buckets] * w[..., None]).sum(2) / w.sum(2)[..., None]   # [P,2,H]

for l in range(L):
    tri = np.asarray(mom[l]).astype(np.float64)                 # [B, tri]
    mu_tr, mu_te = pole_mean(l, TRAIN), pole_mean(l, TEST)
    d_tr = (mu_tr[:, 0] - mu_tr[:, 1]).astype(np.float64)       # [P,H]
    d_te = (mu_te[:, 0] - mu_te[:, 1]).astype(np.float64)
    # leave-one-out shared axis from the training half
    units = d_tr / np.maximum(np.linalg.norm(d_tr, axis=1, keepdims=True), 1e-12)
    total = units.sum(axis=0)
    shared_loo = (total[None, :] - units) / (P - 1)
    shared_loo /= np.maximum(np.linalg.norm(shared_loo, axis=1, keepdims=True), 1e-12)
    # within-pole covariance on the TEST buckets
    M = np.zeros((H, H))
    M[tri_r, tri_c] = tri[TEST].sum(axis=0); M[tri_c, tri_r] = M[tri_r, tri_c]
    gsum = (sums[l][:, :, TEST] * counts[:, :, TEST][..., None]).sum(2).reshape(-1, H).astype(np.float64)
    gcnt = counts[:, :, TEST].sum(2).reshape(-1)
    live = gcnt > 0
    cov = (M - (gsum[live] / gcnt[live, None]).T @ gsum[live]) / gcnt[live].sum()
    cov = (cov + cov.T) * 0.5
    for v, w in enumerate((d_tr, d_tr - (d_tr * shared_loo).sum(1, keepdims=True) * shared_loo)):
        num = (w * d_te).sum(1)
        den = np.sqrt(np.maximum(np.einsum("ph,hk,pk->p", w, cov, w), 1e-12))
        auroc[v, l] = norm.cdf(num / den / np.sqrt(2.0))
    if l % 6 == 0 or l == L - 1:
        print(f"L{l:2d}: raw {np.median(auroc[0,l]):.4f}   minus-shared {np.median(auroc[1,l]):.4f}   "
              f"delta {np.median(auroc[1,l]-auroc[0,l]):+.4f}", flush=True)

atomic_safetensors({"auroc_raw": auroc[0], "auroc_minus_shared": auroc[1]},
                   Path("work/convergence/probe_shared.safetensors"),
                   {"note": "held-out Fisher AUROC, train buckets 01, test buckets 23, LOO shared axis"})
print("done", flush=True)
EOF
