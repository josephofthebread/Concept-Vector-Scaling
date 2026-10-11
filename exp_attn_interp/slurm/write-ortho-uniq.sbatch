#!/bin/bash
#SBATCH --job-name=attn-write-ortho-uniq
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/write-ortho-uniq-%j.out
#SBATCH --error=logs/write-ortho-uniq-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p logs work data
srun .venv/bin/attn-presence
srun .venv/bin/attn-uniqmarkers --siblings "${SIBLINGS:-10}"
srun .venv/bin/attn-write --chunk 0 --vectors-file work/vectors_ortho.safetensors \
    --markers data/markers_unique.json --out-dir work/ortho_uniq
srun .venv/bin/attn-writean --write-dir work/ortho_uniq/write --out work/ortho_uniq/writean.json
