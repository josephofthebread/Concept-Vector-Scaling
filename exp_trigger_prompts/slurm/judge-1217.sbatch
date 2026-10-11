#!/bin/bash
#SBATCH --job-name=judge-1217
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=logs/judge-1217-%A_%a.out
#SBATCH --error=logs/judge-1217-%A_%a.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$ROOT/.venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs work/judge-1217

SHARD=${SHARD:-${SLURM_ARRAY_TASK_ID:?set SHARD or submit with --array}}
SHARDS=${SHARDS:?set SHARDS}
echo "=== [$(date +%H:%M:%S)] judge-1217 shard $SHARD of $SHARDS on $(nvidia-smi -L | head -1) ==="
srun .venv/bin/trigger-judge \
    --steered work/steer/sweep-Qwen_Qwen3-8B-1217.jsonl --features data/features.json \
    --out-dir work/judge-1217 --shard "$SHARD" --shards "$SHARDS" --prompts 8 --tensor-parallel-size 1
