#!/bin/bash
#SBATCH --job-name=hl-examples
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/hl-%j.out
#SBATCH --error=logs/hl-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export PATH="$ROOT/.venv/bin:$PATH"
mkdir -p logs
echo "=== [$(date +%H:%M:%S)] hl-examples on $(hostname) ==="
.venv/bin/python -m exp_lmsys_probe.hl_examples || { echo "hl failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] done: $(ls -la hl_examples.json 2>/dev/null) ==="
