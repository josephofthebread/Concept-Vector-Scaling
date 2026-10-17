#!/bin/bash
#SBATCH --job-name=gcg-steer-judge
#SBATCH --partition=rocky
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p logs work
MODELS=(qwen3_8b llama3_1_8b_instruct gemma2_9b_it)
JUDGES=(finetuned gpt_oss)
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / 2))]}
JUDGE=${JUDGES[$((SLURM_ARRAY_TASK_ID % 2))]}
if [ "$JUDGE" = finetuned ]; then
    EXTRA=(--strongreject qylu4156/strongreject-15k-v1 --strongreject-base google/gemma-2b --templates data/judge_templates.json)
else
    EXTRA=(--gpt-oss openai/gpt-oss-20b)
fi
.venv/bin/python -m exp_gcg steer judge --prepared work/steering/prepared.json --output work/steering \
    --model "$MODEL" --judge "$JUDGE" --device cuda:0 "${EXTRA[@]}"
