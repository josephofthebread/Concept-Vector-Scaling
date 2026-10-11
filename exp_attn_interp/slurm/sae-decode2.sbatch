#!/bin/bash
#SBATCH --job-name=attn-sae-decode2
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/sae-decode2-%j.out
#SBATCH --error=logs/sae-decode2-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun bash -c ".venv/bin/attn-saedecode --model base --wanted work/sae/wanted-base.json --out work/sae/decode2-base.json && .venv/bin/attn-saedecode --model instruct --every 8 --wanted work/sae/wanted-instruct.json --out work/sae/decode2-instruct.json"
