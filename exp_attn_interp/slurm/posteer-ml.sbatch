#!/bin/bash
#SBATCH --job-name=attn-posteer-ml
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=logs/posteer-ml-%j.out
#SBATCH --error=logs/posteer-ml-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PAIRS=20,74,109,173,234,307,384,400,441,518,559,615,633,669,716,818,861,960
ARMS=orig,lang_en,lang_zh,lang_ru,langavg,langst_en
srun bash -c ".venv/bin/attn-posteer --ml --pairs $PAIRS --arms $ARMS --max-new-tokens 96 --alpha 0.5 --out work/posteer-ml/a50 && .venv/bin/attn-posteer --ml --pairs $PAIRS --arms $ARMS --max-new-tokens 96 --alpha 1.0 --out work/posteer-ml/a100 && .venv/bin/attn-posteer --ml --pairs $PAIRS --arms orig --max-new-tokens 96 --alpha 0.0 --out work/posteer-ml/a0"
