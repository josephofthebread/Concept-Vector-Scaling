#!/bin/bash
#SBATCH --job-name=genvectors-paired-extract
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/paired-ext-%j.out
#SBATCH --error=logs/paired-ext-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source .env
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
: "${MI:?set MI=0..5}"
MODELS=("Qwen/Qwen3-8B-Base" "google/gemma-2-9b" "unsloth/Meta-Llama-3.1-8B" \
        "Qwen/Qwen3-8B" "google/gemma-2-9b-it" "unsloth/Meta-Llama-3.1-8B-Instruct")
PRES=("paired_raw" "paired_raw" "paired_raw" "paired_assistant" "paired_assistant" "paired_assistant")
MODEL=${MODELS[$MI]}; PRESENTATION=${PRES[$MI]}
TAG=$(printf '%s--%s--semantic' "${MODEL//\//_}" "$PRESENTATION")
OUT="work/captures/$TAG/shard-000.safetensors"
mkdir -p "$(dirname "$OUT")" logs
[ -s "$OUT" ] && { echo "exists: $OUT"; exit 0; }
echo "=== $MODEL / $PRESENTATION / full data ==="
srun .venv/bin/genvectors-extract --model "$MODEL" --presentation "$PRESENTATION" --minimal \
    --dataset "data/semantic.parquet" --ontology "data/semantic.ontology.json" \
    --shard 0 --shards 1 --out "$OUT" --offline
