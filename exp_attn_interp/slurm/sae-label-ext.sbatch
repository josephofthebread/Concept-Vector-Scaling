#!/bin/bash
#SBATCH --job-name=attn-sae-label-ext
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/sae-label-ext-%j.out
#SBATCH --error=logs/sae-label-ext-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PAIRS=$(python3 -c "import json; d=json.load(open('work/tmp/pairsets.json')); print(','.join(str(x) for x in d['bad']+d['good']))")
srun .venv/bin/attn-saelabel --model Qwen/Qwen3-8B --presentation assistant --rows "${ROWS:-22,24,28}" --pairs "$PAIRS" \
    --contexts 3 --out "${OUT:-work/sae/label-quart.json}"
