#!/bin/bash
#SBATCH --job-name=attn-pojudge2
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=logs/pojudge2-%j.out
#SBATCH --error=logs/pojudge2-%j.err
set -uo pipefail
ATTN=${SLURM_SUBMIT_DIR:-$PWD}
TRIG=${SLURM_SUBMIT_DIR:-$PWD}/../exp_trigger_prompts
cd "$TRIG" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$TRIG/.venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for f in "$ATTN"/work/posteer2/L*/*.jsonl; do
    d=$(basename "$(dirname "$f")"); arm=$(basename "$f" .jsonl)
    [ -s "$f" ] || continue
    mkdir -p "$ATTN/work/pojudge2/$d-$arm"
    echo "=== [$(date +%H:%M:%S)] judging $d $arm ==="
    srun .venv/bin/trigger-judge --steered "$f" --features data/features.json \
        --out-dir "$ATTN/work/pojudge2/$d-$arm" --shard 0 --shards 1 --prompts 8 --tensor-parallel-size 1
done
