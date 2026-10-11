#!/bin/bash
#SBATCH --job-name=gendata-oodconv
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/oodconv-%j.out
#SBATCH --error=logs/oodconv-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK; mkdir -p logs
.venv/bin/python scripts/oodconv.py --tag "${TAG:?}" || { echo "oodconv failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] oodconv $TAG done ==="
