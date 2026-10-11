#!/bin/bash
#SBATCH --job-name=gendata-naming-control
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/naming-%A_%a.out
#SBATCH --error=logs/naming-%A_%a.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
source .env
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
MODELS=("Qwen/Qwen3-8B" "google/gemma-2-9b-it" "unsloth/Meta-Llama-3.1-8B-Instruct")
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / 4))]}
SHARD=$((SLURM_ARRAY_TASK_ID % 4))
TAG=$(printf '%s--named_assistant_decl--semantic' "${MODEL//\//_}")
OUT="work/captures/$TAG/shard-$(printf '%03d' "$SHARD").safetensors"
mkdir -p "$(dirname "$OUT")" logs
if [ -s "$OUT" ]; then
    echo "already complete: $OUT"
    exit 0
fi
srun .venv/bin/gendata-extract \
    --model "$MODEL" --presentation named_assistant \
    --dataset "data/semantic.parquet" --ontology "data/semantic.ontology.json" \
    --shard "$SHARD" --shards 4 --out "$OUT" --offline
