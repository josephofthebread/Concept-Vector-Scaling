#!/bin/bash
#SBATCH --job-name=gendata-convcurve2
#SBATCH --partition=rocky
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/convcurve2-%A_%a.out
#SBATCH --error=logs/convcurve2-%A_%a.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
source .env 2>/dev/null || true
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export PATH="$ROOT/.venv/bin:$PATH"
mkdir -p logs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
SHARD=${SLURM_ARRAY_TASK_ID:?}; SHARDS=${SHARDS:?}
srun .venv/bin/python scripts/convcurve2.py --model "${MODEL:?}" --shard "$SHARD" --shards "$SHARDS" || { echo "convcurve2 failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] convcurve2 shard $SHARD done ==="
