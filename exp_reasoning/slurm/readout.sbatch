#!/bin/bash
#SBATCH --job-name=reasoning-readout
#SBATCH --partition=rocky
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs work/readout
.venv/bin/python -m exp_reasoning.readout --results work/results --out work/readout \
    --vector-dir "${VECTORS:-../exp_genvectors/work/vectors/Qwen_Qwen3-8B--assistant--semantic}" \
    --benchmark gpqa_diamond,math_500 --layer 18 --bins 50 --chunk-tokens 1024 --include-unfinished
