#!/bin/bash
#SBATCH --job-name=attn-posteer-sitanti
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/posteer-sitanti-%j.out
#SBATCH --error=logs/posteer-sitanti-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun bash -c ".venv/bin/attn-posteer --group bad --every 6 --arms sitanti0.5,sitanti1 --layer 22 --alpha 0.5 --max-new-tokens 768 --out work/posteer-sitanti/a0.5 \
  && .venv/bin/python scripts/mergearms.py work/judge-in/sitanti.jsonl sitanti05=work/posteer-sitanti/a0.5/sitanti0.5.jsonl sitanti1=work/posteer-sitanti/a0.5/sitanti1.jsonl sitonly=work/posteer-sit/a0.5/sitonly.jsonl --dup-neg=sitanti05,sitanti1,sitonly"
