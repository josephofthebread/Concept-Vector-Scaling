#!/bin/bash
#SBATCH --job-name=attn-base
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/base-%j.out
#SBATCH --error=logs/base-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
M=Qwen/Qwen3-8B-Base
srun .venv/bin/attn-decomp --model $M --presentation raw --chunk -1 --per-pole 2
srun .venv/bin/attn-decompan --decomp-dir work-Qwen3-8B-Base/decomp --band-start 19 --out work-Qwen3-8B-Base/decompan.json --edgemap none.json
srun .venv/bin/attn-amp --model $M --presentation raw --amps-of Qwen/Qwen3-8B
srun .venv/bin/attn-uni --decomp-dir work-Qwen3-8B-Base/decomp --out work-Qwen3-8B-Base/uni.json
