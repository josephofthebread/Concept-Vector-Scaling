#!/bin/bash
#SBATCH --job-name=gendata-cc3-agg
#SBATCH --partition=cpu-e-quick
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/cc3-agg-%j.out
#SBATCH --error=logs/cc3-agg-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python scripts/convcurve3_agg.py || { echo "aggregation failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] aggregation done ==="
