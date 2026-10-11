#!/bin/bash
#SBATCH --job-name=attn-sit025-good
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/sit025-good-%j.out
#SBATCH --error=logs/sit025-good-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun bash -c ".venv/bin/attn-posteer --group good --every 6 --arms orig --layer 22 --alpha -0.25 --max-new-tokens 768 --batch-size 128 --out work/posteer-sit025/good-m \
  && .venv/bin/attn-posteer --group good --every 6 --arms orig,sit1,sitanti1 --layer 22 --alpha 0.25 --max-new-tokens 768 --batch-size 128 --out work/posteer-sit025/good-p"
