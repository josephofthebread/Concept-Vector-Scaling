#!/bin/bash
#SBATCH --job-name=gendata-psanalysis
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/psanalysis-%j.out
#SBATCH --error=logs/psanalysis-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PSDIR="${1:-work/perstory}"
srun .venv/bin/python - <<'EOF'
import json, os
import numpy as np
from pathlib import Path
from safetensors import safe_open
from exp_gendata.artifacts import atomic_safetensors

def auroc(labels, score):
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    pos = labels.sum()
    if pos == 0 or pos == len(labels): return np.nan
    return (ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * (len(labels) - pos))

rng = np.random.default_rng(0)
psdir = Path(os.environ["PSDIR"])
out, meta_out = {}, {}
for path in sorted(psdir.glob("pair-*.safetensors")):
    with safe_open(str(path), framework="np") as f:
        man = json.loads(f.metadata()["manifest"])
        diff = f.get_tensor("diff").astype(np.float32)  # [rows, layers, hidden]
    pid = man["pair_id"]
    lang = np.array([r["language_id"] for r in man["rows"]])
    gen = np.array([r["generator_id"] for r in man["rows"]])
    mand = lang == 2
    L = diff.shape[1]
    a_mand = np.empty(L, dtype=np.float32); a_gen_mand = np.empty(L, dtype=np.float32)
    a_gen_all = np.empty(L, dtype=np.float32)
    for l in range(L):
        x = diff[:, l]
        a_mand[l] = auroc(mand, x @ (x[mand].mean(0) - x[~mand].mean(0)))
        dist = np.linalg.norm(x[mand] - x[~mand].mean(0), axis=1)
        a_gen_mand[l] = auroc(gen[mand] == 1, dist)
        a_gen_all[l] = auroc(gen == 1, x @ (x[gen == 1].mean(0) - x[gen == 0].mean(0)))
    out[f"p{pid}_auroc_mandarin"] = a_mand
    out[f"p{pid}_auroc_qwen_dist_mandarin"] = a_gen_mand
    out[f"p{pid}_auroc_generator_all"] = a_gen_all
    for l in range(L):
        x = diff[:, l] - diff[:, l].mean(0)
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        basis = vt[:2]
        coords = {1: x @ basis.T}
        perm = rng.permutation(len(x))
        for k in (4, 32):
            groups = perm[: (len(x) // k) * k].reshape(-1, k)
            coords[k] = np.stack([x[g].mean(0) for g in groups]) @ basis.T
        coords[256] = np.stack([x[rng.choice(len(x), 256, replace=False)].mean(0) for _ in range(50)]) @ basis.T
        coords[1024] = (x.mean(0) @ basis.T)[None]
        for k, c in coords.items():
            out[f"p{pid}_L{l}_k{k}"] = c.astype(np.float32)
    meta_out[str(pid)] = {"concept": man["concept"], "antagonist": man["antagonist"],
                          "language": lang.tolist(), "generator": gen.tolist(),
                          "genre": [r["genre_id"] for r in man["rows"]]}
    print("done", pid, flush=True)
atomic_safetensors(out, psdir / f"psanalysis-{psdir.name}.safetensors", {"meta": json.dumps(meta_out)})
EOF
