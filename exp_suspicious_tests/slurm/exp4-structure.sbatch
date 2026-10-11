#!/bin/bash
#SBATCH --job-name=suspicious-exp4-structure
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/exp4-structure-%j.out
#SBATCH --error=logs/exp4-structure-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
POSITION=${POSITION:-role_assistant}
ITEMS=${ITEMS:-0 1 2}
# shellcheck disable=SC2086
srun .venv/bin/python - "$POSITION" $ITEMS <<'EOF'
import json
import sys

import numpy as np
from safetensors import safe_open

kind, items = sys.argv[1], [int(a) for a in sys.argv[2:]]
rng = np.random.default_rng(0)
BINS = np.linspace(0, 1, 21)


def profiles(item: int) -> np.ndarray:
    with safe_open(f"work/exp4/rho/item-{item:04d}.safetensors", framework="np") as handle:
        positions = json.loads(handle.metadata()["positions"])
        index = next(i for i, p in enumerate(positions) if p["kind"] == kind)
        rho = handle.get_tensor("rho")[index].astype(np.float32)
    return rho.reshape(-1, rho.shape[-1])  # [heads, vectors]


def report(name: str, matrix: np.ndarray) -> None:
    centered = matrix - matrix.mean(axis=1, keepdims=True)  # remove the shared component
    unit = centered / np.maximum(np.linalg.norm(centered, axis=0, keepdims=True), 1e-12)
    similarity = np.abs(unit.T @ unit)
    off = similarity[~np.eye(similarity.shape[0], dtype=bool)]
    counts, _ = np.histogram(off, bins=BINS)
    print(f"--- {name}: |cos| mean={off.mean():.3f} median={np.median(off):.3f} "
          f"p90={np.percentile(off, 90):.3f} p99={np.percentile(off, 99):.3f}")
    print("    hist " + " ".join(f"{BINS[i]:.2f}:{100 * c / off.size:5.1f}%" for i, c in enumerate(counts) if c))
    eigenvalues = np.linalg.eigvalsh(unit.T @ unit)[::-1]
    share = eigenvalues / eigenvalues.sum()
    print("    top-15 eigen share " + " ".join(f"{s:.3f}" for s in share[:15]))
    cumulative = np.cumsum(share)
    for target in (0.5, 0.9, 0.99):
        print(f"    components for {int(target * 100)}% variance: {int(np.searchsorted(cumulative, target)) + 1}", end="")
    print()


for item in items:
    matrix = profiles(item)
    print(f"=== item {item}, position {kind}, matrix {matrix.shape} ===")
    report("observed", matrix)
    shuffled = np.take_along_axis(matrix, rng.permuted(
        np.tile(np.arange(matrix.shape[0])[:, None], (1, matrix.shape[1])), axis=0), axis=0)
    report("shuffled control", shuffled)
EOF
