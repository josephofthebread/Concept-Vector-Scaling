#!/bin/bash
#SBATCH --job-name=gendata-decl-conv
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/decl-conv-%j.out
#SBATCH --error=logs/decl-conv-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'PY'
import glob, numpy as np
from safetensors import safe_open
from exp_gendata.convergence import families, subset_vectors, cosine_rows, SUBSETS

DECL={'Qwen3-8B':'Qwen_Qwen3-8B','gemma-2-9b':'google_gemma-2-9b-it','Llama-3.1-8B':'unsloth_Meta-Llama-3.1-8B-Instruct'}
hf_combos=families()['h_f']
for model,slug in DECL.items():
    files=sorted(glob.glob(f'work/captures/{slug}--named_assistant_decl--semantic/shard-*.safetensors'))
    sums=counts=None
    for f in files:
        with safe_open(f,framework='np') as h:
            c=h.get_tensor('counts').astype(np.float64)
            s=h.get_tensor('sums')
            s=(s.astype(np.float64)*np.maximum(c,1)[...,None]) if s.dtype==np.float16 else s.astype(np.float64)
        sums=s if sums is None else sums+s; counts=c if counts is None else counts+c
    L=sums.shape[0]
    angs=[]
    for layer in range(L):
        base={sub:subset_vectors(sums[layer],counts,sub)['diff'] for sub in SUBSETS}
        cos=np.concatenate([cosine_rows(base[a],base[b]) for a,b in hf_combos])
        angs.append(np.degrees(np.arccos(np.clip(np.median(cos),-1,1))))
    print('%-14s named_assistant_decl  median-over-depth angle %.2f deg'%(model,np.median(angs)))
PY
