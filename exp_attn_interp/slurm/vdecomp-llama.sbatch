#!/bin/bash
#SBATCH --job-name=attn-vdecomp-llama
#SBATCH --partition=rocky
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/vdecomp-llama-%j.out
#SBATCH --error=logs/vdecomp-llama-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun .venv/bin/attn-vdecomp --model unsloth/Meta-Llama-3.1-8B-Instruct --out work-Meta-Llama-3.1-8B-Instruct/vdecomp --per-pole 8
