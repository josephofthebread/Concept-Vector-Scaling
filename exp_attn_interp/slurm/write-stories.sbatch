#!/bin/bash
#SBATCH --job-name=attn-write-stories
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/write-stories-%j.out
#SBATCH --error=logs/write-stories-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p logs work
srun .venv/bin/attn-write --stories "${STORIES:-1}" --chunk "${CHUNK:-0}" \
    --vectors-file work/vectors_ortho.safetensors --markers data/markers_unique.json --out-dir work/stories
srun .venv/bin/attn-writean --write-dir work/stories/write --out work/stories/writean.json
