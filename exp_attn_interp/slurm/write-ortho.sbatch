#!/bin/bash
#SBATCH --job-name=attn-write-ortho
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/write-ortho-%j.out
#SBATCH --error=logs/write-ortho-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p logs work
srun .venv/bin/attn-orthovec --siblings "${SIBLINGS:-10}"
srun .venv/bin/attn-write --chunk 0 --vectors-file work/vectors_ortho.safetensors --out-dir work/ortho
srun .venv/bin/attn-writean --write-dir work/ortho/write --out work/ortho/writean.json
