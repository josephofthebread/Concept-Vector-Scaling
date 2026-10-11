#!/bin/bash
#SBATCH --job-name=suspicious-exp2-models
#SBATCH --partition=rocky
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/exp2-models-%A-%a.out
#SBATCH --error=logs/exp2-models-%A-%a.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs work/exp2
MODELS=("google/gemma-2-9b-it" "unsloth/Meta-Llama-3.1-8B-Instruct")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SLUG=${MODEL//\//_}
OUT="work/exp2/exp2-$SLUG.safetensors"
if [ -f "$OUT" ]; then echo "exists: $OUT"; exit 0; fi
srun .venv/bin/suspicious-mcqa --checkpoint "$MODEL" --out "$OUT"
