#!/bin/bash
#SBATCH --job-name=attn-lex2vec
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/lex2vec-%j.out
#SBATCH --error=logs/lex2vec-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun .venv/bin/attn-lex2vec --model Qwen/Qwen3-8B --markers data/markers_presence.json
srun .venv/bin/attn-lex2vec --model unsloth/Meta-Llama-3.1-8B-Instruct --markers data/markers_presence-Meta-Llama-3.1-8B-Instruct.json
srun .venv/bin/attn-lex2vec --model google/gemma-2-9b-it --markers data/markers_presence-gemma-2-9b-it.json
