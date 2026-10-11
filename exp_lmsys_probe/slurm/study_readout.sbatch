#!/bin/bash
#SBATCH --job-name=study-tag
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/study-tag-%j.out
#SBATCH --error=logs/study-tag-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export PATH="$ROOT/.venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORK="${WORK:-work}"; SHARD="${SHARD:-0}"; SHARDS="${SHARDS:-3}"; mkdir -p logs "$WORK"
echo "=== [$(date +%H:%M:%S)] study-tag shard ${SHARD}/${SHARDS} work=$WORK on $(hostname) ==="
.venv/bin/python -m exp_lmsys_probe.tag_readout --work "$WORK" --shard "$SHARD" --shards "$SHARDS" \
    || { echo "study-tag shard ${SHARD} failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] shard ${SHARD} done ==="
