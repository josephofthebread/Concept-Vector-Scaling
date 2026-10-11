#!/bin/bash
#SBATCH --job-name=attn-posteer-sit
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/posteer-sit-%j.out
#SBATCH --error=logs/posteer-sit-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun bash -c "for A in 0.5 -0.5; do .venv/bin/attn-posteer --group bad --every 6 --arms sit0.5,sit1,sitonly --layer 22 --alpha \$A --max-new-tokens 768 --out work/posteer-sit/a\$A; done \
  && .venv/bin/python scripts/mergearms.py work/judge-in/sit.jsonl sit05_p=work/posteer-sit/a0.5/sit0.5.jsonl sit1_p=work/posteer-sit/a0.5/sit1.jsonl sitonly_p=work/posteer-sit/a0.5/sitonly.jsonl sit05_m=work/posteer-sit/a-0.5/sit0.5.jsonl sit1_m=work/posteer-sit/a-0.5/sit1.jsonl sitonly_m=work/posteer-sit/a-0.5/sitonly.jsonl"
