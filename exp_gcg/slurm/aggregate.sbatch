#!/bin/bash
#SBATCH --job-name=gcg-aggregate
#SBATCH --partition=cpu-e-quick
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p logs work
VECTORS="${VECTORS:-../exp_genvectors/work/vectors}"
.venv/bin/python -m exp_gcg aggregate --records data/records.jsonl --vectors "$VECTORS" --output work/suffix
