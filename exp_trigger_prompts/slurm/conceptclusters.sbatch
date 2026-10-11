#!/bin/bash
#SBATCH --job-name=trigger-conceptclusters
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/conceptclusters-%j.out
#SBATCH --error=logs/conceptclusters-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PATH="$ROOT/.venv/bin:$PATH"
mkdir -p logs work
.venv/bin/python -m exp_trigger_prompts.conceptclusters || { echo "conceptclusters failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] done ==="
