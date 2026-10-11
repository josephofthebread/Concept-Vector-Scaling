#!/bin/bash
#SBATCH --job-name=suspicious-exp3-prefilled
#SBATCH --partition=rocky
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/exp3-prefilled-%A-%a.out
#SBATCH --error=logs/exp3-prefilled-%A-%a.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs work/exp3
MODELS=("Qwen/Qwen3-8B" "google/gemma-2-9b-it" "unsloth/Meta-Llama-3.1-8B-Instruct")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SLUG=${MODEL//\//_}
TRANSCRIPTS="work/exp3/transcripts-$SLUG-deepseek.jsonl"
OUT="work/exp3/proj-$SLUG-deepseek.safetensors"
if [ -f "$OUT" ]; then echo "exists: $OUT"; exit 0; fi
[ -f "$TRANSCRIPTS" ] || srun .venv/bin/suspicious-generate --model "$MODEL" \
    --prefills data/prefills-deepseek.json --out "$TRANSCRIPTS"
srun .venv/bin/suspicious-measure --model "$MODEL" --transcripts "$TRANSCRIPTS" --out "$OUT"
