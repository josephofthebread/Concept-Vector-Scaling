#!/bin/bash
#SBATCH --job-name=attn-pojudge
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --time=10:00:00
#SBATCH --output=logs/pojudge-%j.out
#SBATCH --error=logs/pojudge-%j.err
set -uo pipefail
ATTN=${SLURM_SUBMIT_DIR:-$PWD}
TRIG=${SLURM_SUBMIT_DIR:-$PWD}/../exp_trigger_prompts
cd "$TRIG" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$TRIG/.venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for arm in ${ARMS:-orig nomean sproj64 lex0.5 lex1}; do
    [ -s "$ATTN/work/posteer/$arm.jsonl" ] || { echo "missing $arm"; continue; }
    mkdir -p "$ATTN/work/pojudge/$arm"
    echo "=== [$(date +%H:%M:%S)] judging arm $arm ==="
    srun .venv/bin/trigger-judge --steered "$ATTN/work/posteer/$arm.jsonl" --features data/features.json \
        --out-dir "$ATTN/work/pojudge/$arm" --shard 0 --shards 1 --prompts 8 --tensor-parallel-size 1
done
