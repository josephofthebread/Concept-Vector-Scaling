#!/bin/bash
#SBATCH --job-name=gendata-perstory-gemma
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/perstory-gemma-%j.out
#SBATCH --error=logs/perstory-gemma-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source .env
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs work/perstory-gemma
srun .venv/bin/gendata-perstory --model google/gemma-2-9b-it --presentation assistant \
    --pairs jobs/perstory_pairs.json --out work/perstory-gemma
