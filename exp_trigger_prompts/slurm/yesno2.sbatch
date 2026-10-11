#!/bin/bash
#SBATCH --job-name=trigger-yesno2
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/yesno2-%j.out
#SBATCH --error=logs/yesno2-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export PATH="$ROOT/.venv/bin:$PATH"
mkdir -p logs work
.venv/bin/python -m exp_trigger_prompts.yesno2 || { echo "yesno failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] done ==="
