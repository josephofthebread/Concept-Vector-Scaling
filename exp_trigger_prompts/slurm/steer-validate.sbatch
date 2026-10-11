#!/bin/bash
#SBATCH --job-name=steer-validate
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=logs/steer-validate-%j.out
#SBATCH --error=logs/steer-validate-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
mkdir -p logs work/steer

MODEL="Qwen/Qwen3-8B"
TAG="Qwen_Qwen3-8B"

echo "=== [$(date +%H:%M:%S)] calibrate residual norms ==="
.venv/bin/trigger-calibrate --model "$MODEL" \
    --baselines "work/baselines-$TAG.jsonl" --out "work/steer/norms-$TAG.json" --sample 128 \
    || { echo "calibration failed" >&2; exit 1; }

echo "=== [$(date +%H:%M:%S)] steer validation slice ==="
.venv/bin/trigger-steer --model "$MODEL" --prompts work/prompts.jsonl \
    --norms "work/steer/norms-$TAG.json" --out "work/steer/validate-$TAG.jsonl" \
    --shard 0 --shards 1 --limit-concepts 3 --limit-prompts 6 --batch-size 32 \
    || { echo "steer failed" >&2; exit 1; }

echo "=== [$(date +%H:%M:%S)] done ==="
wc -l "work/steer/validate-$TAG.jsonl"
