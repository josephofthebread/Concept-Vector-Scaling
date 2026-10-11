#!/bin/bash
#SBATCH --job-name=attn-posteer-swap
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/posteer-swap-%j.out
#SBATCH --error=logs/posteer-swap-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
ALPHA=${ALPHA:-0.25}
srun bash -c ".venv/bin/attn-posteer --group all --every 16 --arms orig,vrow12,vrow35 --layer 22 --alpha ${ALPHA} --max-new-tokens 96 --out work/posteer-swap/L22 \
  && .venv/bin/attn-posteer --group all --every 16 --arms orig,vrow22 --layer 12 --alpha ${ALPHA} --max-new-tokens 96 --out work/posteer-swap/L12 \
  && mkdir -p work/judge-in && .venv/bin/python scripts/mergearms.py work/judge-in/swap-a025.jsonl v22at22=work/posteer-swap/L22/orig.jsonl v12at22=work/posteer-swap/L22/vrow12.jsonl v35at22=work/posteer-swap/L22/vrow35.jsonl v12at12=work/posteer-swap/L12/orig.jsonl v22at12=work/posteer-swap/L12/vrow22.jsonl --alphas=${ALPHA}"
