#!/bin/bash
#SBATCH --job-name=trigger-zdiag
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/zdiag-%j.out
#SBATCH --error=logs/zdiag-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export PATH="$ROOT/.venv/bin:$PATH"
mkdir -p logs work
for M in "Qwen/Qwen3-8B" "google/gemma-2-9b-it" "unsloth/Meta-Llama-3.1-8B-Instruct"; do
    echo "=== [$(date +%H:%M:%S)] zdiag $M ==="
    .venv/bin/python -m exp_trigger_prompts.zdiag --model "$M" || { echo "zdiag failed for $M" >&2; exit 1; }
done
echo "=== done ==="
