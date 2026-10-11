#!/bin/bash
#SBATCH --job-name=gendata-strip-moments
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/strip-moments-%j.out
#SBATCH --error=logs/strip-moments-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'PY'
import glob, os
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

freed = 0; done = 0; skipped = 0
for path in sorted(glob.glob('work/captures/*/shard-*.safetensors')):
    with safe_open(path, framework='np') as h:
        meta = dict(h.metadata() or {})
        names = list(h.keys())
        if 'moments_tri' not in names:
            skipped += 1; print('skip (no moments_tri):', path, flush=True); continue
        keep = {n: h.get_tensor(n) for n in names if n != 'moments_tri'}
    tmp = path + '.tmp'
    save_file(keep, tmp, metadata=meta)
    with open(tmp, 'rb') as fh:
        os.fsync(fh.fileno())
    # re-read the temp and prove every surviving tensor is byte-identical
    with safe_open(tmp, framework='np') as h2:
        got = set(h2.keys())
        assert got == set(keep), f'{path}: key mismatch {got ^ set(keep)}'
        assert 'moments_tri' not in got, f'{path}: moments_tri survived'
        for n, arr in keep.items():
            r = h2.get_tensor(n)
            assert r.dtype == arr.dtype and r.shape == arr.shape and np.array_equal(r, arr), f'{path}: {n} changed'
    before = os.path.getsize(path)
    os.replace(tmp, path)
    after = os.path.getsize(path); freed += before - after; done += 1
    print('%s: %.1f -> %.1f GiB  (verified %d tensors)' % (path, before/2**30, after/2**30, len(keep)), flush=True)
print('DONE: stripped %d, skipped %d, freed %.1f GiB' % (done, skipped, freed/2**30))
PY
