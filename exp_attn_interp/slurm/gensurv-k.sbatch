#!/bin/bash
#SBATCH --job-name=attn-gensurv-k
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:28:00
#SBATCH --output=logs/gensurv-k-%j.out
#SBATCH --error=logs/gensurv-k-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun .venv/bin/attn-gensurv --every 4 --rows 19,22 --out work/gensurv-k
