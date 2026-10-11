#!/bin/bash
#SBATCH --job-name=attn-posteer-randbag
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
#SBATCH --output=logs/posteer-randbag-%j.out
#SBATCH --error=logs/posteer-randbag-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun bash -c ".venv/bin/attn-posteer --group bad --every 6 --arms sit1,randbag1,mismatch1 --layer 22 --alpha 0.5 --max-new-tokens 768 --batch-size 128 --out work/posteer-randbag/a0.5 \
  && .venv/bin/python scripts/mergearms.py work/judge-in/randbag.jsonl sit1_p=work/posteer-randbag/a0.5/sit1.jsonl randbag1_p=work/posteer-randbag/a0.5/randbag1.jsonl mismatch1_p=work/posteer-randbag/a0.5/mismatch1.jsonl"
