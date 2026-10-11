#!/bin/bash
#SBATCH --job-name=trigger-mljudge
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:30:00
#SBATCH --output=logs/mljudge-%j.out
#SBATCH --error=logs/mljudge-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$ROOT/.venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs work/mljudge
echo "=== [$(date +%H:%M:%S)] mljudge on $(nvidia-smi -L | head -1) ==="
srun .venv/bin/python -m exp_trigger_prompts.mljudge --shard 0 --shards 1 --model "${MODEL:?set MODEL}" || { echo "mljudge failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] mljudge done ==="
