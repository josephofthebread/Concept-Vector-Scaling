#!/bin/bash
#SBATCH --job-name=genvectors-oodplot
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=logs/oodplot-%j.out
#SBATCH --error=logs/oodplot-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs
for fld in lang genre generator; do .venv/bin/python scripts/oodplot.py --field "$fld" || { echo "oodplot $fld failed" >&2; exit 1; }; done
echo "=== [$(date +%H:%M:%S)] oodplot done ==="
