#!/bin/bash
#SBATCH --job-name=attn-posteer-ctx
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/posteer-ctx-%j.out
#SBATCH --error=logs/posteer-ctx-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
ALPHA=${ALPHA:-0.5}
srun .venv/bin/attn-posteer --group bad --every "${EVERY:-6}" --arms orig --layer 22 --alpha "${ALPHA}" --context "${CONTEXT:-guidance}" \
    --max-new-tokens "${MAXTOK:-768}" --out work/posteer-ctx/a"${ALPHA}"
