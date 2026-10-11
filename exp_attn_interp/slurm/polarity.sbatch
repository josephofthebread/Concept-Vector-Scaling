#!/bin/bash
#SBATCH --job-name=attn-polarity
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=logs/polarity-%j.out
#SBATCH --error=logs/polarity-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun .venv/bin/attn-polarity --model Qwen/Qwen3-8B --out work/polarity-qwen.json
srun .venv/bin/attn-polarity --model unsloth/Meta-Llama-3.1-8B-Instruct --out work/polarity-llama.json
srun .venv/bin/attn-polarity --model google/gemma-2-9b-it --out work/polarity-gemma.json
srun .venv/bin/attn-polarity --model Qwen/Qwen3-8B-Base --out work/polarity-qwen-base.json
