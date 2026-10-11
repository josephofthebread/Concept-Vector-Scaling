#!/bin/bash
#SBATCH --job-name=suspicious-pilot15
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/pilot15-%j.out
#SBATCH --error=logs/pilot15-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs
echo "===== instruct ====="
srun .venv/bin/suspicious-mcqa --model instruct --pilot 16
echo "===== base ====="
srun .venv/bin/suspicious-mcqa --model base --pilot 16
