#!/bin/bash
#SBATCH --job-name=attn-reliab
#SBATCH --partition=cpu-e-quick
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=logs/reliab-%j.out
#SBATCH --error=logs/reliab-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
srun .venv/bin/python scripts/reliab.py | tee work/tmp/reliab.out
