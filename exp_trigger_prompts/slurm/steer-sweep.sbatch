#!/bin/bash
#SBATCH --job-name=steer-sweep
#SBATCH --partition=rocky
#SBATCH --constraint=type_h|type_e|type_f
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/steer-sweep-%A_%a.out
#SBATCH --error=logs/steer-sweep-%A_%a.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs work/steer

MODEL="Qwen/Qwen3-8B"
TAG="Qwen_Qwen3-8B"
SHARD=${SHARD:-${SLURM_ARRAY_TASK_ID:?set SHARD or submit with --array}}
SHARDS=${SHARDS:?set SHARDS}
if [ ! -s "work/steer/norms-$TAG.json" ]; then
    echo "missing work/steer/norms-$TAG.json (run calibration first)" >&2
    exit 1
fi
echo "=== [$(date +%H:%M:%S)] shard $SHARD of $SHARDS on $(nvidia-smi -L | head -1) ==="
srun .venv/bin/trigger-steer \
    --model "$MODEL" --prompts work/prompts.jsonl --norms "work/steer/norms-$TAG.json" \
    --out "work/steer/sweep-$TAG-shard-$SHARD.jsonl" --shard "$SHARD" --shards "$SHARDS"
