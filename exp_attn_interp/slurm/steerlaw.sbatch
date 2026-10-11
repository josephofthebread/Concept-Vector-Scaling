#!/bin/bash
#SBATCH --job-name=attn-steerabl
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/steerlaw-%j.out
#SBATCH --error=logs/steerlaw-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs work
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun .venv/bin/attn-steerabl --model "${MODEL:-Qwen/Qwen3-8B}" --layer 21 --alpha 0.25 --law --n-pairs 300 --prompts-per-pair 2 --new-tokens 80 --out work/steerlaw
