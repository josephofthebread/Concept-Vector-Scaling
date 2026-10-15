#!/bin/bash
#SBATCH --job-name=genvectors-strip
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/strip-%j.out
#SBATCH --error=logs/strip-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
srun .venv/bin/python - <<'EOF'
import json
from pathlib import Path
from safetensors import safe_open
from exp_genvectors.artifacts import atomic_safetensors

KEEP = ("eigenvalues", "reliability_raw", "reliability_whitened")
for path in sorted(Path("work/vectors").glob("*/diagnostics.safetensors")):
    with safe_open(str(path), framework="np") as handle:
        metadata = dict(handle.metadata() or {})
        names = list(handle.keys())
        if set(names) <= set(KEEP):
            print(f"already stripped: {path}")
            continue
        kept = {name: handle.get_tensor(name) for name in KEEP}
    before = path.stat().st_size
    atomic_safetensors(kept, path, metadata)
    print(f"{path}: {before / 2**30:.2f} -> {path.stat().st_size / 2**30:.3f} GiB")
EOF
