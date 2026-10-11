#!/bin/bash
#SBATCH --job-name=gendata-ps70b
#SBATCH --partition=rocky
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=3
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/ps70b-%j.out
#SBATCH --error=logs/ps70b-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source .env
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
MODEL="${PSMODEL:-unsloth/Meta-Llama-3.1-70B}"
OUT="${PSOUT:-work/perstory-warmth-70b}"
mkdir -p logs "$OUT"
srun .venv/bin/gendata-perstory --model "$MODEL" --presentation raw \
    --pairs jobs/warmth6.json --out "$OUT"
