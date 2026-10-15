#!/bin/bash
#SBATCH --job-name=genvectors-oodrepsb
#SBATCH --partition=rocky
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/oodrepsb-%A_%a.out
#SBATCH --error=logs/oodrepsb-%A_%a.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export PATH="$ROOT/.venv/bin:$PATH"
mkdir -p logs
SHARD=${SLURM_ARRAY_TASK_ID:?}; SHARDS=${SHARDS:?}
srun .venv/bin/python scripts/oodrepsb.py --model "${MODEL:?}" --shard "$SHARD" --shards "$SHARDS" --cap "${CAP:-200}" --stride "${STRIDE:-2}" || { echo "oodrepsb failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] oodrepsb shard $SHARD done ==="
