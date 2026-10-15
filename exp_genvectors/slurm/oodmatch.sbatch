#!/bin/bash
#SBATCH --job-name=genvectors-oodmatch
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/oodmatch-%j.out
#SBATCH --error=logs/oodmatch-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs
.venv/bin/python scripts/oodmatch.py --tag "${TAG:-Qwen3-8B}" || { echo "oodmatch failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] oodmatch done ==="
