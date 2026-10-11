#!/bin/bash
#SBATCH --job-name=attn-polejudge
#SBATCH --partition=test
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/polejudge-%j.out
#SBATCH --error=logs/polejudge-%j.err
set -uo pipefail
ATTN=${SLURM_SUBMIT_DIR:-$PWD}
TRIG=${SLURM_SUBMIT_DIR:-$PWD}/../exp_trigger_prompts
cd "$ATTN" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p work/pojudge-pole
srun "$TRIG/.venv/bin/python" scripts/polejudge.py --inp "$ATTN/work/judge-in/${INP:-pole}.jsonl" --out "$ATTN/work/pojudge-pole/${INP:-pole}.jsonl" \
    --features "$TRIG/data/features.json"
