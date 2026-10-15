#!/bin/bash
#SBATCH --job-name=genvectors-paired-vectors
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/paired-vec-%j.out
#SBATCH --error=logs/paired-vec-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
MODELS=("Qwen/Qwen3-8B-Base" "google/gemma-2-9b" "unsloth/Meta-Llama-3.1-8B" \
        "Qwen/Qwen3-8B" "google/gemma-2-9b-it" "unsloth/Meta-Llama-3.1-8B-Instruct")
PRES=("paired_raw" "paired_raw" "paired_raw" "paired_assistant" "paired_assistant" "paired_assistant")
for i in "${!MODELS[@]}"; do
    TAG=$(printf '%s--%s--semantic' "${MODELS[$i]//\//_}" "${PRES[$i]}")
    mkdir -p "work/vectors/$TAG" logs
    echo "=== $TAG ==="
    TAG="$TAG" srun .venv/bin/python - <<'PY'
import os, glob, numpy as np
from pathlib import Path
from exp_genvectors.artifacts import read_capture, atomic_safetensors, canonical_json
from exp_genvectors.vecmath import means
tag = os.environ['TAG']
files = sorted(glob.glob(f'work/captures/{tag}/shard-*.safetensors'))
assert files, tag
manifest = None; sums = counts = None
for f in files:
    m, t = read_capture(f)
    manifest = manifest or m
    s = t['sums'].astype(np.float64); c = t['counts'].astype(np.float64)
    sums = s if sums is None else sums + s
    counts = c if counts is None else counts + c
mu = means(sums, counts)                    # [layer, pair, side, hidden]
diff = (mu[:, :, 0] - mu[:, :, 1]).astype(np.float32)
assert np.isfinite(diff).all(), f'{tag}: non-finite diff'
meta = {'manifest': canonical_json({'model': manifest['model'], 'presentation': manifest['presentation'],
        'dataset': manifest['dataset'], 'layers': int(manifest['layers']),
        'hidden_size': int(manifest['hidden_size']), 'pairs': diff.shape[1],
        'vector': 'diff', 'axes': ['layer', 'pair', 'hidden']})}
atomic_safetensors({'diff': diff}, Path(f'work/vectors/{tag}/diff.safetensors'), meta)
print('wrote work/vectors/%s/diff.safetensors %s' % (tag, diff.shape))
PY
done
