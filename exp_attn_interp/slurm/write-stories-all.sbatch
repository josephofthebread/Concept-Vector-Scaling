#!/bin/bash
#SBATCH --job-name=attn-stories-all
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --array=0-16%12
#SBATCH --output=logs/stories-all-%A-%a.out
#SBATCH --error=logs/stories-all-%A-%a.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p logs work
srun .venv/bin/attn-write --stories 1 --chunk "$SLURM_ARRAY_TASK_ID" \
    --vectors-file work/vectors_ortho.safetensors --markers data/markers_unique.json --out-dir work/stories
