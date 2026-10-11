#!/bin/bash
#SBATCH --job-name=gendata-perstory-llama
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/perstory-llama-%j.out
#SBATCH --error=logs/perstory-llama-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source .env
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs work/perstory-llama
srun .venv/bin/gendata-perstory --model unsloth/Meta-Llama-3.1-8B-Instruct --presentation assistant \
    --pairs jobs/perstory_pairs.json --out work/perstory-llama
