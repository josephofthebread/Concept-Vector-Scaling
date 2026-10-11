#!/bin/bash
#SBATCH --job-name=attn-posteer-parts2
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/posteer-parts2-%j.out
#SBATCH --error=logs/posteer-parts2-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
ALPHA=${ALPHA:-0.5}
srun .venv/bin/attn-posteer --group all --every 16 --arms orig,p_whole,p_attn,p_mlp,p_xattn,p_attnhi,p_mlphi --layer 22 \
    --alpha "${ALPHA}" --max-new-tokens 96 --out work/posteer-parts2/a"${ALPHA}"
