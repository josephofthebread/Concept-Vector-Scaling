#!/bin/bash
#SBATCH --job-name=attn-figs-cdce
#SBATCH --partition=cpu-e-quick
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=logs/figs-cdce-%j.out
#SBATCH --error=logs/figs-cdce-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg
srun .venv/bin/python scripts/figs_cdce.py
