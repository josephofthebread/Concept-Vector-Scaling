#!/bin/bash
#SBATCH --job-name=gendata-langscores-qwen
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/langscores-qwen-%j.out
#SBATCH --error=logs/langscores-qwen-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source .env
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs work/langscores-qwen
srun .venv/bin/gendata-langscores --model Qwen/Qwen3-8B --presentation assistant \
    --capture work/captures/Qwen_Qwen3-8B--assistant--semantic/shard-000.safetensors \
    --out work/langscores-qwen
