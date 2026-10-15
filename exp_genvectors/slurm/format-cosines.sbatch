#!/bin/bash
#SBATCH --job-name=genvectors-formatcos
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=logs/formatcos-%j.out
#SBATCH --error=logs/formatcos-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python scripts/paperfigs/format_cosines.py
