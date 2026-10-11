#!/bin/bash
#SBATCH --job-name=trigger-scopegen
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/scopegen-%j.out
#SBATCH --error=logs/scopegen-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export PATH="$ROOT/.venv/bin:$PATH"
mkdir -p logs work
rm -rf work/scope-data
.venv/bin/python -m exp_trigger_prompts.scopegen || { echo "scopegen failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] done: $(find work/scope-data -name '*.bin' | wc -l) blobs, $(du -sh work/scope-data | cut -f1) ==="
