#!/bin/bash
#SBATCH --job-name=genvectors-format-instability
#SBATCH --partition=cpu-e-quick
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/format-instability-%j.out
#SBATCH --error=logs/format-instability-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python scripts/paperfigs/format_instability.py
