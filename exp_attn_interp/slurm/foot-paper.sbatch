#!/bin/bash
#SBATCH --job-name=attn-foot-paper
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/foot-paper-%j.out
#SBATCH --error=logs/foot-paper-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun .venv/bin/python scripts/sweepfilter.py work/tmp/sweep-row22.jsonl row=22 alphas=0.5,-0.5 every="${EVERY:-3}" prompts="${PROMPTS:-8}"
srun .venv/bin/python scripts/footprint.py --records work/tmp/sweep-row22.jsonl --labels work/sae/label-all.json \
    --rows 22 --out work/footprint/paper.jsonl
