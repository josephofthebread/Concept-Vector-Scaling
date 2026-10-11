#!/bin/bash
#SBATCH --job-name=hl-readout
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/hl-%j.out
#SBATCH --error=logs/hl-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export PATH="$ROOT/.venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORK="${WORK:-work}"; CORPUS="${CORPUS:-lmsys}"; MODEL="${MODEL:-Qwen/Qwen3-8B}"
SHARD="${SHARD:-0}"; SHARDS="${SHARDS:-3}"; mkdir -p logs "$WORK"
SEL="${SEL:-examples_sel.json}"; PREFIX="${PREFIX:-hl}"; MERGED="${MERGED:-highlights.json}"
echo "=== [$(date +%H:%M:%S)] hl-readout ${CORPUS} shard ${SHARD}/${SHARDS} work=$WORK sel=$SEL prefix=$PREFIX on $(hostname) ==="
.venv/bin/python -m exp_lmsys_probe.hl_readout --work "$WORK" --corpus "$CORPUS" --model "$MODEL" \
    --shard "$SHARD" --shards "$SHARDS" --sel "$SEL" --prefix "$PREFIX" --merged "$MERGED" \
    || { echo "hl-readout ${CORPUS} shard ${SHARD} failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] ${CORPUS} shard ${SHARD} done ==="
