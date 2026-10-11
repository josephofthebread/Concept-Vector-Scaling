#!/bin/bash
#SBATCH --job-name=attn-posteer-amp
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
#SBATCH --output=logs/posteer-amp-%j.out
#SBATCH --error=logs/posteer-amp-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
P25=0,43,86,129,172,215,258,301,344,387,430,473,516,559,602,645,688,731,774,817,860,903,946,989,1032
OUT=work/posteer-amp
for L in 16 24; do
    for A in 0.5 -0.5; do
        srun .venv/bin/attn-posteer --pairs $P25 --arms orig --layer $L --alpha $A --max-new-tokens 96 --out $OUT --tag _amp2 --ablate "19:28=2,19:29=2,19:31=2"
        srun .venv/bin/attn-posteer --pairs $P25 --arms orig --layer $L --alpha $A --max-new-tokens 96 --out $OUT --tag _amp0 --ablate "19:28,19:29,19:31"
        srun .venv/bin/attn-posteer --pairs $P25 --arms orig --layer $L --alpha $A --max-new-tokens 96 --out $OUT --tag _rnd2 --ablate "19:3=2,19:4=2,19:18=2"
    done
done
srun .venv/bin/attn-posteer --pairs $P25 --arms orig --layer 22 --alpha 0 --max-new-tokens 96 --out $OUT --tag _amp2 --ablate "19:28=2,19:29=2,19:31=2"
srun .venv/bin/attn-posteer --pairs $P25 --arms orig --layer 22 --alpha 0 --max-new-tokens 96 --out $OUT --tag _amp0 --ablate "19:28,19:29,19:31"
