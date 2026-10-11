#!/bin/bash
#SBATCH --job-name=gendata-convcurve-agg
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/convcurve-agg-%j.out
#SBATCH --error=logs/convcurve-agg-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs work/convcurve
.venv/bin/python scripts/convcurve_agg.py --tag "${TAG:?}" || { echo "convcurve_agg failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] convcurve-agg done ==="
