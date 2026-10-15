#!/bin/bash
#SBATCH --job-name=genvectors-lexprobedisc
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=logs/lexprobedisc-%j.out
#SBATCH --error=logs/lexprobedisc-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs work
.venv/bin/python scripts/lexprobe_disc.py --model "${MODEL:?set MODEL}" || { echo "lexprobe_disc failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] done ==="
