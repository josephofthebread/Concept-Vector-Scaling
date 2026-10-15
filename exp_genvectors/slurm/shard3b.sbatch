#!/bin/bash
#SBATCH --job-name=genvectors-3b
#SBATCH --partition=rocky
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/s3b-%j.out
#SBATCH --error=logs/s3b-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source .env
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
OUT=work/captures/Llama-3.2-3B--raw--semantic300
mkdir -p logs "$OUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
srun .venv/bin/genvectors-extract \
    --model unsloth/Llama-3.2-3B --presentation raw \
    --dataset data/semantic.parquet --ontology data/semantic.ontology.json \
    --pair-subset jobs/pairs300.json --buckets 0 \
    --minimal --workers 4 --token-budget 16384 \
    --out "$OUT/shard-000.safetensors" --offline
