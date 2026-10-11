#!/bin/bash
#SBATCH --job-name=attn-sae-label-all
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/sae-label-all-%j.out
#SBATCH --error=logs/sae-label-all-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PAIRS=$(python3 -c "print(','.join(str(i) for i in range(1036)))")
srun .venv/bin/attn-saelabel --model Qwen/Qwen3-8B --presentation assistant --rows "${ROWS:-22}" --pairs "$PAIRS" \
    --per-pole "${PERPOLE:-8}" --contexts 3 --out "${OUT:-work/sae/label-all.json}"
