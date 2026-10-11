#!/bin/bash
#SBATCH --job-name=suspicious-exp1-measure
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-9%6
#SBATCH --output=logs/exp1-measure-%A-%a.out
#SBATCH --error=logs/exp1-measure-%A-%a.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs work/exp1
ARMS=(main roleplay)
PROBES=(angry calm happy sad scared)
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID / 5))]}
PROBE=${PROBES[$((SLURM_ARRAY_TASK_ID % 5))]}
OUT="work/exp1/exp1-$ARM-$PROBE.safetensors"
if [ -f "$OUT" ]; then echo "exists: $OUT"; exit 0; fi
srun .venv/bin/suspicious-measure --transcripts work/transcripts.jsonl \
    --arm "$ARM" --probe "$PROBE" --out "$OUT"
