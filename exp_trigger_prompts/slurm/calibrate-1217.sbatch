#!/bin/bash
#SBATCH --job-name=calib-1217
#SBATCH --partition=rocky
#SBATCH --constraint=type_h|type_e|type_f
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/calib-1217-%j.out
#SBATCH --error=logs/calib-1217-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs work/steer
echo "=== [$(date +%H:%M:%S)] calibrate 12..17 on $(nvidia-smi -L | head -1) ==="
srun .venv/bin/trigger-calibrate --model "Qwen/Qwen3-8B" \
    --baselines work/baselines-Qwen_Qwen3-8B.jsonl \
    --out work/steer/norms-Qwen_Qwen3-8B-1217.json
