#!/bin/bash
#SBATCH --job-name=attn-posteer2
#SBATCH --partition=rocky
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/posteer2-%j.out
#SBATCH --error=logs/posteer2-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun .venv/bin/attn-posteer --group all --every 8 --layer 22 --arms orig,vrow24,vrow19,vrow35 --out work/posteer2/L22
srun .venv/bin/attn-posteer --group all --every 8 --layer 24 --arms orig,vrow22,vrow19 --out work/posteer2/L24
srun .venv/bin/attn-posteer --group all --every 8 --layer 19 --arms orig,vrow22,vrow24 --out work/posteer2/L19
