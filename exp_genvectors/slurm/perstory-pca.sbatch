#!/bin/bash
#SBATCH --job-name=genvectors-pspca
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/pspca-%j.out
#SBATCH --error=logs/pspca-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'EOF'
import json
import numpy as np
from pathlib import Path
from safetensors import safe_open
from exp_genvectors.artifacts import atomic_safetensors

LAYERS = tuple(range(37))
out, meta = {}, {}
for path in sorted(Path("work/perstory").glob("pair-*.safetensors")):
    with safe_open(str(path), framework="np") as f:
        man = json.loads(f.metadata()["manifest"])
        diff = f.get_tensor("diff").astype(np.float32)  # [rows, layers, hidden]
    pid = man["pair_id"]
    for layer in LAYERS:
        x = diff[:, layer]
        x = x - x.mean(axis=0)
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        coords = u[:, :2] * s[:2]
        var = (s ** 2) / (s ** 2).sum()
        out[f"p{pid}_L{layer}"] = coords.astype(np.float32)
        out[f"p{pid}_L{layer}_var"] = var[:10].astype(np.float32)
    meta[str(pid)] = {"concept": man["concept"], "antagonist": man["antagonist"],
                      "genre": [r["genre_id"] for r in man["rows"]],
                      "language": [r["language_id"] for r in man["rows"]],
                      "generator": [r["generator_id"] for r in man["rows"]]}
    print("done", pid, flush=True)
atomic_safetensors(out, Path("work/perstory/pca2d.safetensors"), {"meta": json.dumps(meta)})
EOF
