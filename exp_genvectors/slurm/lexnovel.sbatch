#!/bin/bash
#SBATCH --job-name=genvectors-lexnovel
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/lexnovel-%j.out
#SBATCH --error=logs/lexnovel-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs work
.venv/bin/python scripts/lexnovel.py --model "${MODEL:?set MODEL}" || { echo "lexnovel failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] done ==="
