#!/bin/bash
#SBATCH --job-name=attn-basepos-llama
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=logs/basepos-llama-%j.out
#SBATCH --error=logs/basepos-llama-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun .venv/bin/attn-basepos --model unsloth/Meta-Llama-3.1-8B-Instruct --rows 11,15,31 --sweep sweep-unsloth_Meta-Llama-3.1-8B-Instruct-shard-0.jsonl --storyproj work-Meta-Llama-3.1-8B-Instruct/storyproj --out work-Meta-Llama-3.1-8B-Instruct/basepos.npz
