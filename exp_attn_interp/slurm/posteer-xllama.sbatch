#!/bin/bash
#SBATCH --job-name=attn-posteer-xllama
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/posteer-xllama-%j.out
#SBATCH --error=logs/posteer-xllama-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun bash -c ".venv/bin/attn-posteer --group all --every 32 --arms xllama --max-new-tokens 96 --alpha 0.25 --out work/posteer-words/a25 && .venv/bin/attn-posteer --group all --every 32 --arms xllama --max-new-tokens 96 --alpha 0.5 --out work/posteer-words/a50"
