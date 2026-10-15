#!/bin/bash
#SBATCH --job-name=genvectors-oodplotb
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=logs/oodplotb-%j.out
#SBATCH --error=logs/oodplotb-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK; mkdir -p logs
for fld in lang genre generator; do .venv/bin/python scripts/oodplotb.py --field "$fld" || { echo "oodplotb $fld failed" >&2; exit 1; }; done
echo "=== [$(date +%H:%M:%S)] oodplotb done ==="
