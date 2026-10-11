#!/bin/bash
#SBATCH --job-name=analyze-gemma
#SBATCH --partition=cpu-e-quick
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=logs/analyze-gemma-%j.out
#SBATCH --error=logs/analyze-gemma-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=2
echo "=== [$(date +%H:%M:%S)] analyze gemma judge scores ==="
.venv/bin/python scripts/analyze_judge.py --in-dir work/judge-gemma --out-dir work/judge-gemma
