#!/bin/bash
#SBATCH --job-name=gendata-70b
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=logs/e70b-%j.out
#SBATCH --error=logs/e70b-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source .env
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
OUT=work/captures/unsloth_Meta-Llama-3.1-70B--raw--semantic300
mkdir -p logs "$OUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
for s in 0 1 2 3; do
    if [ -s "$OUT/shard-00$s.safetensors" ]; then echo "shard $s already done"; continue; fi
    CUDA_VISIBLE_DEVICES=$((2*s)),$((2*s+1)) \
    .venv/bin/gendata-extract \
        --model unsloth/Meta-Llama-3.1-70B --presentation raw \
        --dataset data/semantic.parquet --ontology data/semantic.ontology.json \
        --pair-subset jobs/pairs300.json --buckets 0 \
        --shard $s --shards 4 --split-model --minimal --workers 4 --token-budget 16384 \
        --out "$OUT/shard-00$s.safetensors" --offline \
        > logs/e70b-shard$s.log 2>&1 &
    echo "launched shard $s on GPUs $((2*s)),$((2*s+1))"
done
wait
echo "all shards finished"
ls -lh "$OUT"
