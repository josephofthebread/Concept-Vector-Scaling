#!/bin/bash
#SBATCH --job-name=gendata-lsauroc
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/lsauroc-%j.out
#SBATCH --error=logs/lsauroc-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'EOF'
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
from exp_gendata.artifacts import atomic_safetensors

def auroc_cols(labels, scores):
    """AUROC of every column of scores [n, k] against one boolean label vector."""
    order = np.argsort(scores, axis=0)
    ranks = np.empty_like(order, dtype=np.float64)
    n, k = scores.shape
    rows = np.arange(1, n + 1, dtype=np.float64)[:, None]
    np.put_along_axis(ranks, order, np.broadcast_to(rows, (n, k)), axis=0)
    pos = labels.sum()
    if pos == 0 or pos == n:
        return np.full(k, np.nan)
    return (ranks[labels].sum(axis=0) - pos * (pos + 1) / 2) / (pos * (n - pos))

paths = sorted(Path("work/langscores-qwen").glob("pair-*.safetensors"))
mand, qdist, gen, pids = [], [], [], []
for path in paths:
    with safe_open(str(path), framework="np") as f:
        man = json.loads(f.metadata()["manifest"])
        s = f.get_tensor("scores").astype(np.float32)      # [rows, layers, 3]
    lang = np.array([r["language_id"] for r in man["rows"]])
    gid = np.array([r["generator_id"] for r in man["rows"]])
    is_m = lang == 2
    mand.append(auroc_cols(is_m, s[:, :, 0]))                        # mandarin projection
    qdist.append(auroc_cols(gid[is_m] == 1, s[is_m][:, :, 1]))       # distance, mandarin rows only
    gen.append(auroc_cols(gid == 1, s[:, :, 2]))                     # generator projection
    pids.append(man["pair_id"])
    if len(pids) % 200 == 0:
        print(f"{len(pids)}/{len(paths)}", flush=True)
out = {"auroc_mandarin": np.stack(mand).astype(np.float32),          # [pair, layer]
       "auroc_qwen_dist_mandarin": np.stack(qdist).astype(np.float32),
       "auroc_generator": np.stack(gen).astype(np.float32),
       "pair_id": np.array(pids, dtype=np.int32)}
atomic_safetensors(out, Path("work/langscores-qwen/auroc.safetensors"), {"note": "full corpus per-pair AUROC"})
for name in ("auroc_mandarin", "auroc_qwen_dist_mandarin", "auroc_generator"):
    a = out[name]
    L = a.shape[1]
    print(f"{name}: L0 median {np.nanmedian(a[:,0]):.3f} | mid {np.nanmedian(a[:,L//2]):.3f} | "
          f"L{L-11} median {np.nanmedian(a[:,L-11]):.3f} | >0.9 at mid: {(a[:,L//2]>0.9).sum()}/{a.shape[0]}", flush=True)
EOF
