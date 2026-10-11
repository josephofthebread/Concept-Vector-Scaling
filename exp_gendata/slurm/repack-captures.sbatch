#!/bin/bash
#SBATCH --job-name=gendata-repack
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/repack-%j.out
#SBATCH --error=logs/repack-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'EOF'
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from exp_gendata.artifacts import atomic_safetensors, pack_capture, read_capture, unpack_capture
from exp_gendata.vecmath import cosine, means, whiten, within_covariance

MIN_COSINE_DIFF = 0.9999
MIN_COSINE_WHITE = 0.999

for path in sorted(Path("work/captures").glob("*/shard-*.safetensors")):
    with safe_open(str(path), framework="np") as handle:
        metadata = dict(handle.metadata() or {})
        names = list(handle.keys())
    if "moments_tri" in names:
        print(f"already packed: {path}", flush=True)
        continue
    manifest, original = read_capture(path)
    packed = pack_capture(original, mean16=manifest["dataset"] != "syntax")
    restored = unpack_capture(dict(packed))

    layer = int(manifest["layers"]) // 2
    counts = original["counts"]
    truth_diff = np.squeeze(np.diff(means(original["sums"], counts), axis=2) * -1, axis=2)[layer]
    packed_diff = np.squeeze(np.diff(means(restored["sums"], counts), axis=2) * -1, axis=2)[layer]
    diff_cos = cosine(truth_diff, packed_diff).min()

    group_counts = counts.reshape(-1, counts.shape[-1]).sum(axis=1)
    truth_cov = within_covariance(
        original["moments"][layer].sum(axis=0),
        original["sums"][layer].sum(axis=2).reshape(-1, truth_diff.shape[-1]),
        group_counts,
    )
    packed_cov = within_covariance(
        restored["moments"][layer].sum(axis=0),
        restored["sums"][layer].astype(np.float64).sum(axis=2).reshape(-1, truth_diff.shape[-1]),
        group_counts,
    )
    white_cos = cosine(
        whiten(truth_diff.astype(np.float64), truth_cov, 0.05),
        whiten(packed_diff.astype(np.float64), packed_cov, 0.05),
    ).min()
    if diff_cos < MIN_COSINE_DIFF or white_cos < MIN_COSINE_WHITE:
        raise SystemExit(f"{path}: verification failed, diff cos {diff_cos:.6f}, whitened cos {white_cos:.6f}")

    manifest["storage"] = "mean16-tri32" if manifest["dataset"] != "syntax" else "tri32"
    metadata["manifest"] = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    before = path.stat().st_size
    atomic_safetensors(packed, path, metadata)
    print(
        f"{path}: {before / 2**30:.1f} -> {path.stat().st_size / 2**30:.1f} GiB, "
        f"diff cos {diff_cos:.6f}, whitened cos {white_cos:.6f}",
        flush=True,
    )
EOF
