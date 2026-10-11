#!/bin/bash
#SBATCH --job-name=gendata-prepare
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/prepare-%j.out
#SBATCH --error=logs/prepare-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
source .env
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p logs data
.venv/bin/gendata-prepare --out data
.venv/bin/gendata-cache-models
