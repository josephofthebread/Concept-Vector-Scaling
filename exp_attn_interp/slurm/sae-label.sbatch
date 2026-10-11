#!/bin/bash
#SBATCH --job-name=attn-sae-label
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/sae-label-%j.out
#SBATCH --error=logs/sae-label-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
if [ "${MODEL:-base}" = "base" ]; then
    srun .venv/bin/attn-saelabel --model Qwen/Qwen3-8B-Base --presentation raw --features work/sae/decode-base.json --wanted work/sae/wanted-base.json --out work/sae/label-base.json
else
    srun .venv/bin/attn-saelabel --model Qwen/Qwen3-8B --presentation assistant --features work/sae/decode-instruct.json --wanted work/sae/wanted-instruct.json --out work/sae/label-instruct.json
fi
