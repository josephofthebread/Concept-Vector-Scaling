#!/bin/bash
#SBATCH --job-name=genvectors-perstory
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/perstory-%j.out
#SBATCH --error=logs/perstory-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source .env
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs work/perstory
srun .venv/bin/genvectors-perstory --model Qwen/Qwen3-8B --presentation assistant \
    --pairs jobs/perstory_pairs.json --out work/perstory
