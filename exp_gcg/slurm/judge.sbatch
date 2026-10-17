#!/bin/bash
#SBATCH --job-name=gcg-judge
#SBATCH --partition=rocky
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p logs work
VECTORS="${VECTORS:-../exp_genvectors/work/vectors}"
.venv/bin/python -m exp_gcg judge --records data/records.jsonl --vectors "$VECTORS" --output work/suffix \
    --templates data/judge_templates.json --device cuda:0
