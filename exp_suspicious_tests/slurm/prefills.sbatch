#!/bin/bash
#SBATCH --job-name=suspicious-prefills
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/prefills-%j.out
#SBATCH --error=logs/prefills-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs data
srun .venv/bin/suspicious-generate \
    --prefill-model deepseek-ai/deepseek-llm-7b-chat \
    --prefills-out data/prefills-deepseek.json
