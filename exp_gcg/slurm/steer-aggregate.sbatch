#!/bin/bash
#SBATCH --job-name=gcg-steer-aggregate
#SBATCH --partition=cpu-e-quick
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p logs work
.venv/bin/python -m exp_gcg steer aggregate --prepared work/steering/prepared.json --output work/steering
.venv/bin/python -m exp_gcg steer figures --output work/steering --figures-output work/figures
