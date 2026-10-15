#!/bin/bash
#SBATCH --job-name=genvectors-langscores
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/langscores-%j.out
#SBATCH --error=logs/langscores-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source .env
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs work/langscores
srun .venv/bin/genvectors-langscores --model unsloth/Meta-Llama-3.1-8B-Instruct --presentation assistant \
    --capture work/captures/unsloth_Meta-Llama-3.1-8B-Instruct--assistant--semantic/shard-000.safetensors \
    --out work/langscores
