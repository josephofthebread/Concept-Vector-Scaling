#!/bin/bash
#SBATCH --job-name=attn-footprint
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/footprint-%j.out
#SBATCH --error=logs/footprint-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun .venv/bin/python scripts/footprint.py --records "${RECORDS:-work/posteer-foot/orig.jsonl}" \
    --labels "${LABELS:-work/sae/label-instruct.json}" --rows "${ROWS:-22,24,28,35}" --out "${OUT:-work/footprint/foot.jsonl}"
