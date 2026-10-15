#!/bin/bash
#SBATCH --job-name=genvectors-grouptransfer
#SBATCH --partition=cpu-e-quick
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/grouptransfer-%j.out
#SBATCH --error=logs/grouptransfer-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
mkdir -p work/tmp/paperfigs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python scripts/paperfigs/group_transfer.py
