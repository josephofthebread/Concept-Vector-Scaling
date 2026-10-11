#!/bin/bash
#SBATCH --job-name=calib-llama
#SBATCH --partition=test
#SBATCH --constraint=type_e
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=logs/calib-llama-%j.out
#SBATCH --error=logs/calib-llama-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs work/steer
echo "=== [$(date +%H:%M:%S)] calibrate llama 11..21 on $(nvidia-smi -L | head -1) ==="
srun .venv/bin/trigger-calibrate --model "unsloth/Meta-Llama-3.1-8B-Instruct" \
    --baselines work/baselines-unsloth_Meta-Llama-3.1-8B-Instruct.jsonl \
    --out work/steer/norms-unsloth_Meta-Llama-3.1-8B-Instruct.json
