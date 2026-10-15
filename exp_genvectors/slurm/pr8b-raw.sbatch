#!/bin/bash
#SBATCH --job-name=genvectors-pr8b-raw
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/pr8braw-%j.out
#SBATCH --error=logs/pr8braw-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'EOF'
import numpy as np
from pathlib import Path
from exp_genvectors.artifacts import read_capture

MANDARIN = 2
SRC = Path("work/captures/unsloth_Meta-Llama-3.1-8B--raw--semantic/shard-000.safetensors")
_, t = read_capture(SRC)

sums, counts = t["sums"], t["counts"].astype(np.float64)
L, P, _, B, H = sums.shape
pole = sums.sum(3) / np.maximum(counts.sum(2), 1)[None, ..., None]
diff_all = (pole[:, :, 0] - pole[:, :, 1]).astype(np.float32)

lang, lang_ct = t["language_sum"], t["language_count"].astype(np.float64)
keep = [i for i in range(lang_ct.shape[2]) if i != MANDARIN]
pole_nm = lang[:, :, :, keep].sum(3) / np.maximum(lang_ct[:, :, keep].sum(2), 1)[None, ..., None]
diff_nm = (pole_nm[:, :, 0] - pole_nm[:, :, 1]).astype(np.float32)

print(f"{SRC}\n{P} concepts, {L} layers, hidden {H}, {B} buckets, "
      f"{counts.sum(2).mean():.0f} stories per pole ({lang_ct[:, :, keep].sum(2).mean():.0f} non-Mandarin)")


def pr(x):
    lam = np.linalg.svd(x, compute_uv=False) ** 2
    cum = np.cumsum(lam) / lam.sum()
    return (float(lam.sum() ** 2 / (lam ** 2).sum()), int(np.searchsorted(cum, .5) + 1),
            int(np.searchsorted(cum, .9) + 1), float(lam[0] / lam.sum()))


def spectrum(V, l):
    u = V[l] / np.maximum(np.linalg.norm(V[l], axis=1, keepdims=True), 1e-9)
    return pr(u - u.mean(0)), pr(u)


rng = np.random.default_rng(0)
g = rng.standard_normal((P, H)).astype(np.float32)
g /= np.linalg.norm(g, axis=1, keepdims=True)
iso = pr(g - g.mean(0))[0]
print(f"isotropic reference for {P} random unit vectors in {H} dims: PR = {iso:.1f}\n")

print(f"{'layer':>5s} | {'PR':>7s} {'n50':>4s} {'n90':>4s} {'PC1%':>6s}  (all languages)"
      f" | {'PR':>7s} {'n50':>4s} {'n90':>4s} {'PC1%':>6s}  (no Mandarin)")
for l in range(L):
    ca, ra = spectrum(diff_all, l)
    cn, rn = spectrum(diff_nm, l)
    print(f"L{l:<4d} | {ca[0]:7.1f} {ca[1]:4d} {ca[2]:4d} {ra[3]*100:5.1f}%"
          f"                   | {cn[0]:7.1f} {cn[1]:4d} {cn[2]:4d} {rn[3]*100:5.1f}%")
print("\ndone", flush=True)
EOF
