#!/bin/bash
#SBATCH --job-name=gcg-generate
#SBATCH --partition=rocky
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p logs work
VECTORS="${VECTORS:-../exp_genvectors/work/vectors}"
MODELS=(qwen3_8b llama3_1_8b_instruct gemma2_9b_it)
.venv/bin/python -m exp_gcg generate --records data/records.jsonl --vectors "$VECTORS" --output work/suffix \
    --model "${MODELS[$SLURM_ARRAY_TASK_ID]}" --templates data/judge_templates.json --device cuda:0
