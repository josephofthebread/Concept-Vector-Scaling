#!/bin/bash
#SBATCH --job-name=attn-posteer-sit025
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/posteer-sit025-%j.out
#SBATCH --error=logs/posteer-sit025-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun bash -c "for G in good bad; do .venv/bin/attn-posteer --group \$G --every 6 --arms orig --layer 22 --alpha -0.25 --max-new-tokens 768 --batch-size 128 --out work/posteer-sit025/\$G-m \
  && .venv/bin/attn-posteer --group \$G --every 6 --arms orig,sit1,sitanti1 --layer 22 --alpha 0.25 --max-new-tokens 768 --batch-size 128 --out work/posteer-sit025/\$G-p; done \
  && .venv/bin/python scripts/mergearms.py work/judge-in/sit025.jsonl good_orig_p=work/posteer-sit025/good-p/orig.jsonl good_sit1=work/posteer-sit025/good-p/sit1.jsonl good_sitanti1=work/posteer-sit025/good-p/sitanti1.jsonl good_orig_m=work/posteer-sit025/good-m/orig.jsonl bad_orig_p=work/posteer-sit025/bad-p/orig.jsonl bad_sit1=work/posteer-sit025/bad-p/sit1.jsonl bad_sitanti1=work/posteer-sit025/bad-p/sitanti1.jsonl bad_orig_m=work/posteer-sit025/bad-m/orig.jsonl --dup-neg=good_sitanti1,bad_sitanti1"
