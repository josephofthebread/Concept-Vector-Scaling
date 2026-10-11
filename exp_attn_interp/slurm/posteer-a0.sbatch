#!/bin/bash
#SBATCH --job-name=attn-posteer-a0
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/posteer-a0-%j.out
#SBATCH --error=logs/posteer-a0-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun bash -c ".venv/bin/attn-posteer --group all --every 16 --arms orig --layer 22 --alpha 0 --max-new-tokens 96 --out work/posteer-parts2/a0 \
  && .venv/bin/python scripts/mergearms.py work/judge-in/parts-a0.jsonl unsteered96=work/posteer-parts2/a0/orig.jsonl"
