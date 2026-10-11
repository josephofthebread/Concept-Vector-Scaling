#!/bin/bash
#SBATCH --job-name=attn-reach-rows
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/reach-rows-%j.out
#SBATCH --error=logs/reach-rows-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
for ROW in ${ROWS:-12 16 19 24}; do
    srun .venv/bin/attn-reach --layer "$ROW" --alphas 0.25,0.5 --batch-size 8 --every "${EVERY:-16}" --out work/reach
done
