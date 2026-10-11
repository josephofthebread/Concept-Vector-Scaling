#!/bin/bash
#SBATCH --job-name=gendata-warmth6
#SBATCH --partition=gpu-ef-quick
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=type_e|type_f
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/warmth6-%j.out
#SBATCH --error=logs/warmth6-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source .env
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs "$OUTDIR"
nvidia-smi --query-gpu=index,name --format=csv,noheader
srun .venv/bin/gendata-perstory \
    --model "$MODEL" --presentation "$PRESENTATION" \
    --pairs jobs/warmth6.json --out "$OUTDIR" --token-budget 16384
