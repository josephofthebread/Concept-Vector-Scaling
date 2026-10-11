#!/bin/bash
#SBATCH --job-name=study-judge
#SBATCH --partition=test
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/study-judge-%j.out
#SBATCH --error=logs/study-judge-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH="$ROOT/.venv/bin:$PATH"
WORK="${WORK:-work}"; SHARD="${SHARD:-0}"; SHARDS="${SHARDS:-3}"; mkdir -p logs "$WORK"
echo "=== [$(date +%H:%M:%S)] study-judge shard ${SHARD}/${SHARDS} work=$WORK on $(hostname) ==="
.venv/bin/python -m exp_lmsys_probe.judge --study --work "$WORK" --shard "$SHARD" --shards "$SHARDS" \
    ${MAXLEN:+--max-model-len "$MAXLEN"} ${GPUMEM:+--gpu-memory-utilization "$GPUMEM"} \
    || { echo "study-judge shard ${SHARD} failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] shard ${SHARD} done ==="
