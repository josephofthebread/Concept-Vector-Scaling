#!/bin/bash
#SBATCH --job-name=gcg-steer-prepare
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
.venv/bin/python -m exp_gcg steer prepare --records data/records.jsonl --vectors "$VECTORS" --original work/suffix \
    --benign "${BENIGN:?set BENIGN to the 100 Alpaca prompts}" --output work/steering
