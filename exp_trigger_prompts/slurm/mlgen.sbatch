#!/bin/bash
#SBATCH --job-name=trigger-mlgen
#SBATCH --partition=rocky
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/mlgen-%A_%a.out
#SBATCH --error=logs/mlgen-%A_%a.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export PATH="$ROOT/.venv/bin:$PATH"
mkdir -p logs work/mlgen
SHARD=${SLURM_ARRAY_TASK_ID:?submit with --array}
SHARDS=${SHARDS:?set SHARDS}
echo "=== [$(date +%H:%M:%S)] mlgen shard $SHARD/$SHARDS on $(nvidia-smi -L | head -1) ==="
srun .venv/bin/python -m exp_trigger_prompts.mlgen --shard "$SHARD" --shards "$SHARDS" --model "${MODEL:?set MODEL}" || { echo "mlgen failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] mlgen shard $SHARD done ==="
