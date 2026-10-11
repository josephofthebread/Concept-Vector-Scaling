#!/bin/bash
#SBATCH --job-name=steer-1217
#SBATCH --partition=rocky
#SBATCH --constraint=type_h|type_e|type_f
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/steer-1217-%A_%a.out
#SBATCH --error=logs/steer-1217-%A_%a.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs work/steer

MODEL="Qwen/Qwen3-8B"
TAG="Qwen_Qwen3-8B-1217"
NORMS="work/steer/norms-Qwen_Qwen3-8B-1217.json"
SHARD=${SHARD:-${SLURM_ARRAY_TASK_ID:?set SHARD or submit with --array}}
SHARDS=${SHARDS:?set SHARDS}
if [ ! -s "$NORMS" ]; then
    echo "missing $NORMS (run slurm/calibrate-1217.sbatch first)" >&2
    exit 1
fi
echo "=== [$(date +%H:%M:%S)] steer-1217 shard $SHARD of $SHARDS on $(nvidia-smi -L | head -1) ==="
srun .venv/bin/trigger-steer \
    --model "$MODEL" --prompts work/prompts.jsonl --norms "$NORMS" \
    --out "work/steer/sweep-$TAG-shard-$SHARD.jsonl" --shard "$SHARD" --shards "$SHARDS"
