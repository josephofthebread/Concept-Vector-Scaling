#!/bin/bash
#SBATCH --job-name=genvectors-oodcurveb
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/oodcurveb-%j.out
#SBATCH --error=logs/oodcurveb-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs
.venv/bin/python scripts/oodcurveb.py --tag "${TAG:?}" || { echo "oodcurveb failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] oodcurveb done ==="
