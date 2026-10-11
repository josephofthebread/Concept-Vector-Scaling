#!/bin/bash
#SBATCH --job-name=attn-posteer-sitgood2
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
#SBATCH --output=logs/posteer-sitgood2-%j.out
#SBATCH --error=logs/posteer-sitgood2-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun bash -c ".venv/bin/attn-posteer --group good --every 6 --arms sitanti1,sitanti0.5 --layer 22 --alpha 0.5 --max-new-tokens 768 --batch-size 128 --out work/posteer-sitgood/anti \
  && .venv/bin/python scripts/mergearms.py work/judge-in/sitgood.jsonl orig_p=work/posteer-sitgood/a0.5/orig.jsonl sit1_p=work/posteer-sitgood/a0.5/sit1.jsonl sitonly_p=work/posteer-sitgood/a0.5/sitonly.jsonl orig_m=work/posteer-sitgood/a-0.5/orig.jsonl sit1_m=work/posteer-sitgood/a-0.5/sit1.jsonl sitonly_m=work/posteer-sitgood/a-0.5/sitonly.jsonl sitanti1=work/posteer-sitgood/anti/sitanti1.jsonl sitanti05=work/posteer-sitgood/anti/sitanti0.5.jsonl --dup-neg=sitanti1,sitanti05"
