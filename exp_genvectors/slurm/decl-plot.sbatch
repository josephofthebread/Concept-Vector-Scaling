#!/bin/bash
#SBATCH --job-name=genvectors-decl-plot
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/decl-plot-%j.out
#SBATCH --error=logs/decl-plot-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'PY'
import glob, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from safetensors import safe_open
from exp_genvectors.convergence import families, subset_vectors, cosine_rows, SUBSETS
hf=families()['h_f']

def curve_from_scores(slug):
    with safe_open(f'work/convergence/{slug}--semantic.safetensors',framework='np') as h:
        a=h.get_tensor('h_f').astype(np.float64)
    return np.degrees(np.arccos(np.clip(np.median(a.reshape(a.shape[0],-1),axis=1),-1,1)))

def curve_from_shards(slug):
    sums=counts=None
    for f in sorted(glob.glob(f'work/captures/{slug}--named_assistant_decl--semantic/shard-*.safetensors')):
        with safe_open(f,framework='np') as h:
            c=h.get_tensor('counts').astype(np.float64); s=h.get_tensor('sums')
            s=(s.astype(np.float64)*np.maximum(c,1)[...,None]) if s.dtype==np.float16 else s.astype(np.float64)
        sums=s if sums is None else sums+s; counts=c if counts is None else counts+c
    out=[]
    for L in range(sums.shape[0]):
        base={sub:subset_vectors(sums[L],counts,sub)['diff'] for sub in SUBSETS}
        out.append(np.degrees(np.arccos(np.clip(np.median(np.concatenate([cosine_rows(base[a],base[b]) for a,b in hf])),-1,1))))
    return np.array(out)

MODELS={'Qwen3-8B':('Qwen_Qwen3-8B','Qwen_Qwen3-8B-Base'),
        'gemma-2-9b':('google_gemma-2-9b-it','google_gemma-2-9b'),
        'Llama-3.1-8B':('unsloth_Meta-Llama-3.1-8B-Instruct','unsloth_Meta-Llama-3.1-8B')}
COL={'raw':'#888','named_raw':'#e6a817','user':'#4aa564','assistant':'#2a78d6',
     'named_assistant':'#d1495b','named_assistant_decl':'#111'}
fig,axes=plt.subplots(1,3,figsize=(16,4.6),sharey=True)
for ax,(model,(inst,base)) in zip(axes,MODELS.items()):
    tags=[('raw',f'{base}--raw'),('named_raw',f'{base}--named_raw'),('user',f'{inst}--user'),
          ('assistant',f'{inst}--assistant'),('named_assistant',f'{inst}--named_assistant')]
    for name,slug in tags:
        c=curve_from_scores(slug); d=np.arange(len(c))/(len(c)-1)
        ax.plot(d,c,color=COL[name],lw=1.8,label=name)
    c=curve_from_shards(inst); d=np.arange(len(c))/(len(c)-1)
    ax.plot(d,c,color=COL['named_assistant_decl'],lw=2.4,ls='--',label='named_assistant_decl')
    ax.set_title(model,fontsize=11); ax.set_xlabel('fractional depth')
    ax.grid(color='#e6e6ef',lw=.5); ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_visible(False)
axes[0].set_ylabel('median angle: 0.5 vs 1.0 diff vectors (deg)'); axes[0].invert_yaxis()
axes[2].legend(frameon=False,fontsize=8,loc='upper right')
fig.suptitle('h_f (0.5 vs 1.0) with new declarative named_assistant overlaid (dashed black)',fontsize=12)
fig.tight_layout(); fig.savefig('work/figures/convergence-h_f-decl.svg')
print('wrote work/figures/convergence-h_f-decl.svg')
PY
