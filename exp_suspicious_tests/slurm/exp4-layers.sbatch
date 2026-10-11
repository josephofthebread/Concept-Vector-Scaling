#!/bin/bash
#SBATCH --job-name=suspicious-exp4-layers
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/exp4-layers-%j.out
#SBATCH --error=logs/exp4-layers-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
ITEM=${ITEM:-0}
POSITION=${POSITION:-role_assistant}
srun .venv/bin/python - "$ITEM" "$POSITION" <<'EOF'
import json
import sys

import numpy as np
from safetensors import safe_open

item, kind = int(sys.argv[1]), sys.argv[2]
with safe_open(f"work/exp4/rho/item-{item:04d}.safetensors", framework="np") as handle:
    positions = json.loads(handle.metadata()["positions"])
    index = next(i for i, p in enumerate(positions) if p["kind"] == kind)
    rho = handle.get_tensor("rho")[index].astype(np.float32)  # [blocks, heads, pairs]
pairs = json.loads(open("../exp_gendata/data/semantic.ontology.json").read())["pairs"]
pairs.sort(key=lambda p: int(p["pair_id"]))
name = lambda k: f'{pairs[k]["concept"]} || {pairs[k]["antagonist"]}'
blocks, heads, npairs = rho.shape

print(f"=== item {item}, position {kind}: every layer, its strongest couplings ===")
print(f"{'L':>3} {'max|r|':>7} {'med|r|':>7} {'ratio':>6}  top concepts (head:rho)")
for block in range(blocks):
    layer = rho[block]                      # [heads, pairs]
    flat = np.abs(layer).ravel()
    top = np.argsort(-flat)[:3]
    entries = []
    for f in top:
        head, k = divmod(int(f), npairs)
        entries.append(f"H{head}:{layer[head, k]:+.2f} {name(k)[:42]}")
    peak, med = float(np.abs(layer).max()), float(np.median(np.abs(layer)))
    print(f"{block:>3} {peak:7.3f} {med:7.3f} {peak / max(med, 1e-9):6.1f}  " + " | ".join(entries))
EOF
