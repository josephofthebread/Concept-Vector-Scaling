#!/bin/bash
#SBATCH --job-name=attn-gensurv-llama
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:28:00
#SBATCH --output=logs/gensurv-llama-%j.out
#SBATCH --error=logs/gensurv-llama-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun .venv/bin/attn-gensurv --model unsloth/Meta-Llama-3.1-8B-Instruct --every 32 --rows 8,11,13,14,15,17,19,21,24 --out work-Meta-Llama-3.1-8B-Instruct/gensurv
