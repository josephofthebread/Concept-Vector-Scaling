#!/bin/bash
#SBATCH --job-name=judge-baseline
#SBATCH --partition=test
#SBATCH --gpus=1
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:30:00
#SBATCH --output=logs/judge-baseline-%j.out
#SBATCH --error=logs/judge-baseline-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$ROOT/.venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs work/judge-baseline

TAG=${TAG:?set TAG}
SHARD=${SHARD:?set SHARD}
SHARDS=${SHARDS:?set SHARDS}
SWEEP="work/steer/sweep-$TAG.jsonl"
[ -s "$SWEEP" ] || { echo "missing $SWEEP" >&2; exit 1; }
LIMARG=""; [ "${LIMIT:-0}" -gt 0 ] && LIMARG="--limit $LIMIT"

echo "=== [$(date +%H:%M:%S)] baseline judge $TAG shard $SHARD/$SHARDS on $(nvidia-smi -L | head -1) ==="
# shellcheck disable=SC2086
srun .venv/bin/python -m exp_trigger_prompts.judge_baseline \
    --sweep "$SWEEP" --features data/features.json \
    --out-dir "work/judge-baseline/$TAG" --shard "$SHARD" --shards "$SHARDS" \
    --prompts 8 --tensor-parallel-size 1 $LIMARG
echo "=== [$(date +%H:%M:%S)] done shard $SHARD/$SHARDS ==="
