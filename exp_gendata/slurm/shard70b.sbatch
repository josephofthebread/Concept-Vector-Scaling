#!/bin/bash
#SBATCH --job-name=gendata-70b-shard
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/s70b-%j.out
#SBATCH --error=logs/s70b-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source .env
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

SHARD="${SHARD:-0}"; SHARDS="${SHARDS:-1}"; MAXROWS="${MAXROWS:-0}"
OUT="${OUTDIR:-work/captures/unsloth_Meta-Llama-3.1-70B--raw--semantic}"
EXTRA=""
[ -n "${PAIRS:-}" ] && EXTRA="$EXTRA --pair-subset $PAIRS"
[ -n "${BUCKETS:-}" ] && EXTRA="$EXTRA --buckets $BUCKETS"

mkdir -p logs "$OUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
if [ -s "$OUT/shard-00$SHARD.safetensors" ]; then echo "shard $SHARD already complete"; exit 0; fi
# shellcheck disable=SC2086
srun .venv/bin/gendata-extract \
    --model unsloth/Meta-Llama-3.1-70B --presentation raw \
    --dataset data/semantic.parquet --ontology data/semantic.ontology.json \
    --shard "$SHARD" --shards "$SHARDS" --max-rows "$MAXROWS" \
    --split-model --minimal --workers 4 --token-budget "${TOKENBUDGET:-8192}" $EXTRA \
    --out "$OUT/shard-00$SHARD.safetensors" --offline
