#!/bin/bash
#SBATCH --job-name=analysis-llama
#SBATCH --partition=cpu-e-quick
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/analysis-llama-%j.out
#SBATCH --error=logs/analysis-llama-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=2
echo "=== [$(date +%H:%M:%S)] analyze Llama judge ==="
.venv/bin/python scripts/analyze_judge.py --in-dir work/judge-llama --out-dir work/judge-llama
