#!/bin/bash
#SBATCH --job-name=genvectors-conv-defs
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/convdefs-%A_%a.out
#SBATCH --error=logs/convdefs-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
TAG=$(ls work/captures | grep -- --semantic$ | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
mkdir -p logs
srun .venv/bin/genvectors-convergence --tag "$TAG" \
    --definitions concept_centered antagonist_centered whitened_diff whitened_concept_centered whitened_antagonist_centered
