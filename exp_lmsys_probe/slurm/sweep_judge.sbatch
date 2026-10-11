#!/bin/bash
#SBATCH --job-name=sweep-judge
#SBATCH --partition=rocky
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/sweep-judge-%A_%a.out
#SBATCH --error=logs/sweep-judge-%A_%a.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH="$ROOT/.venv/bin:$PATH"
WORK="${WORK:-work}"; OUTDIR="${OUTDIR:-$WORK/span_sweep}"; DESCMODE="${DESCMODE:-full}"; SHARDS="${SHARDS:-6}"
HLFILE="${HLFILE:-$WORK/highlights_all.json}"; SCALE="${SCALE:-strict}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p logs "$OUTDIR"
echo "=== [$(date +%H:%M:%S)] sweep-judge hl=$HLFILE shard $SHARD/$SHARDS desc=$DESCMODE scale=$SCALE out=$OUTDIR on $(hostname) ==="
.venv/bin/python -m exp_lmsys_probe.span_judge --sweep "$HLFILE" --shard "$SHARD" \
    --shards "$SHARDS" --out "$OUTDIR" --desc-mode "$DESCMODE" --scale "$SCALE" --efforts low \
    || { echo "sweep-judge shard $SHARD failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] shard $SHARD done ==="
