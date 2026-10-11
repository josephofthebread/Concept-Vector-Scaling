#!/bin/bash
#SBATCH --job-name=attn-pojudge-test
#SBATCH --partition=test
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/pojudge-test-%j.out
#SBATCH --error=logs/pojudge-test-%j.err
set -uo pipefail
TRIG=${SLURM_SUBMIT_DIR:-$PWD}/../exp_trigger_prompts
cd "$TRIG" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$TRIG/.venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
srun .venv/bin/trigger-judge --steered "$STEERED" --features data/features.json --out-dir "$OUT" --shard 0 --shards 1 \
    --prompts "${PROMPTS:-1000}" --tensor-parallel-size 1
