#!/bin/bash
#SBATCH --job-name=gendata-extract
#SBATCH --partition=rocky
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/extract-%A_%a.out
#SBATCH --error=logs/extract-%A_%a.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
source .env
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
read -r MODEL PRESENTATION DATASET SHARD SHARDS < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" jobs/extract.tsv)
TAG=$(printf '%s--%s--%s' "${MODEL//\//_}" "$PRESENTATION" "$DATASET")
OUT="work/captures/$TAG/shard-$(printf '%03d' "$SHARD").safetensors"
mkdir -p "$(dirname "$OUT")" logs
if [ -s "$OUT" ]; then
    echo "already complete: $OUT"
    exit 0
fi
srun .venv/bin/gendata-extract \
    --model "$MODEL" --presentation "$PRESENTATION" \
    --dataset "data/$DATASET.parquet" --ontology "data/$DATASET.ontology.json" \
    --shard "$SHARD" --shards "$SHARDS" --out "$OUT" --offline
