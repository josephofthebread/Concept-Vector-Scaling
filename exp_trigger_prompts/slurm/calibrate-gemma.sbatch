#!/bin/bash
#SBATCH --job-name=calib-gemma
#SBATCH --partition=rocky
#SBATCH --constraint=type_h|type_e|type_f
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/calib-gemma-%j.out
#SBATCH --error=logs/calib-gemma-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs work/steer
echo "=== [$(date +%H:%M:%S)] calibrate gemma 14..28 on $(nvidia-smi -L | head -1) ==="
srun .venv/bin/trigger-calibrate --model "google/gemma-2-9b-it" \
    --baselines work/baselines-google_gemma-2-9b-it.jsonl \
    --out work/steer/norms-google_gemma-2-9b-it.json
