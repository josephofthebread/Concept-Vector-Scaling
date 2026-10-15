#!/bin/bash
#SBATCH --job-name=suspicious-exp4-readout
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/exp4-readout-%j.out
#SBATCH --error=logs/exp4-readout-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs
ITEM=${ITEM:-0}
POSITION=${POSITION:-role_assistant}
srun .venv/bin/python - "$ITEM" "$POSITION" <<'EOF'
import json
import sys
from collections import Counter

import numpy as np
from safetensors import safe_open

item, kind = int(sys.argv[1]), sys.argv[2]
with safe_open(f"work/exp4/rho/item-{item:04d}.safetensors", framework="np") as handle:
    positions = json.loads(handle.metadata()["positions"])
    index = next(i for i, p in enumerate(positions) if p["kind"] == kind)
    rho = handle.get_tensor("rho")[index].astype(np.float32)  # [blocks, heads, pairs]
transcript = json.loads(open(f"work/exp4/transcripts/item-{item:04d}.json").read())
pairs = json.loads(open("../exp_genvectors/data/semantic.ontology.json").read())["pairs"]
pairs.sort(key=lambda p: int(p["pair_id"]))
name = lambda k: f'{pairs[k]["concept"]} || {pairs[k]["antagonist"]}'

blocks, heads, npairs = rho.shape
M = rho.reshape(blocks * heads, npairs)
print(f"=== item {item}, position {kind} @ token {positions[index]['position']} "
      f"of {len(transcript['input_ids'])}, class_id {transcript['class_id']} ===")
print("USER:", transcript["text"][:700].replace("\n", " | "))
print("REPLY:", transcript["reply"][:400].replace("\n", " | "))
print(f"matrix {M.shape}, |rho| percentiles "
      + " ".join(f"p{q}={np.percentile(np.abs(M), q):.3f}" for q in (50, 90, 99, 99.9)) +
      f" max={np.abs(M).max():.3f}")

order = np.argsort(-np.abs(M).ravel())[:60]
print("=== top 60 couplings ===")
for flat in order:
    row, k = divmod(int(flat), npairs)
    block, head = divmod(row, heads)
    print(f"L{block:2d} H{head:2d}  rho={M[row, k]:+.3f}  {name(k)}"[:150])

print("=== heads carrying those couplings, with their own top-5 concepts ===")
for row, count in Counter(int(f) // npairs for f in order).most_common(10):
    block, head = divmod(row, heads)
    profile = M[row]
    print(f"L{block:2d} H{head:2d} (in top-60 {count}x, max|rho|={np.abs(profile).max():.3f}, "
          f"median|rho|={np.median(np.abs(profile)):.3f})")
    for k in np.argsort(-np.abs(profile))[:5]:
        print(f"      {profile[k]:+.3f}  {name(k)}"[:140])

print("=== concepts in those couplings, with the heads they attach to ===")
for k, count in Counter(int(f) % npairs for f in order).most_common(8):
    column = M[:, k]
    tops = np.argsort(-np.abs(column))[:5]
    where = " ".join(f"L{r // heads}H{r % heads}:{column[r]:+.2f}" for r in tops)
    print(f"{name(k)[:70]:70s} {where}")
EOF
