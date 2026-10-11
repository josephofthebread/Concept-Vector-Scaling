#!/bin/bash
#SBATCH --job-name=trigger-crossrepro
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/crossrepro-%j.out
#SBATCH --error=logs/crossrepro-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PATH="$ROOT/.venv/bin:$PATH"
mkdir -p logs work
.venv/bin/python -m exp_trigger_prompts.crossrepro || { echo "crossrepro failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] done ==="
