#!/bin/bash
#SBATCH --job-name=suspicious-exp4-sanity
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/exp4-sanity-%j.out
#SBATCH --error=logs/exp4-sanity-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs
srun .venv/bin/python - <<'EOF'
import json
from pathlib import Path
import numpy as np
from safetensors import safe_open

pairs = json.loads(open('../exp_gendata/data/semantic.ontology.json').read())['pairs']
pairs.sort(key=lambda p: int(p['pair_id']))

files = sorted(Path('work/exp4/rho').glob('item-*.safetensors'))
sum_grid = None; n_grid = 0
kind_absmean = {}; kind_n = {}
zero_rows = 0; total_rows = 0
lo, hi = 0.0, 0.0
for path in files:
    with safe_open(str(path), framework='np') as h:
        rho = h.get_tensor('rho').astype(np.float32)     # [P, 36, 32, 1036]
        meta = json.loads(h.metadata()['positions'])
    kinds = [m['kind'] for m in meta]
    valid = np.abs(rho).sum(axis=(1, 2, 3)) > 0
    zero_rows += int((~valid).sum()); total_rows += len(valid)
    lo = min(lo, float(rho.min())); hi = max(hi, float(rho.max()))
    for k in set(kinds):
        rows = [i for i, (kk, v) in enumerate(zip(kinds, valid)) if kk == k and v]
        if rows:
            kind_absmean[k] = kind_absmean.get(k, 0) + float(np.abs(rho[rows]).mean())
            kind_n[k] = kind_n.get(k, 0) + 1
    grid_rows = [i for i, (kk, v) in enumerate(zip(kinds, valid)) if kk == 'grid' and v]
    s = rho[grid_rows].mean(axis=0)
    sum_grid = s if sum_grid is None else sum_grid + s
    n_grid += 1
mean_m = sum_grid / n_grid
print(f'files={len(files)} zero rows: {zero_rows}/{total_rows}')
print(f'rho range: min={lo:.3f} max={hi:.3f}')
print('mean |rho| by kind:', {k: round(kind_absmean[k] / kind_n[k], 3) for k in sorted(kind_absmean)})

M = mean_m.reshape(36 * 32, 1036)
s = np.linalg.svd(M - M.mean(axis=1, keepdims=True), compute_uv=False)
print('SVD energy shares (lmsys, grid):', np.round(s[:5]**2 / (s**2).sum(), 3))

flat = np.abs(M).ravel(); top = np.argsort(-flat)[:12]
print('=== top-12 couples (mean over 64 items, grid) ===')
for idx in top:
    hb, k = divmod(idx, 1036); b, hd = divmod(hb, 32)
    print(f'L{b:2d} H{hd:2d} rho={M[hb, k]:+.3f} {pairs[k]["concept"]} || {pairs[k]["antagonist"]}'[:150])

for b, hd, label in [(3, 0, 'question-asking (trashtalk)'), (7, 22, 'rule-adherence'), (8, 23, 'deference')]:
    profile = M[b * 32 + hd]
    topk = np.argsort(-np.abs(profile))[:3]
    names = '; '.join(f'{profile[k]:+.2f} {pairs[k]["concept"]}||{pairs[k]["antagonist"]}' for k in topk)
    print(f'known head L{b} H{hd} ({label}): {names}'[:200])
EOF
