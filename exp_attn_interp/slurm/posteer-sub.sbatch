#!/bin/bash
#SBATCH --job-name=attn-posteer-sub
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/posteer-sub-%j.out
#SBATCH --error=logs/posteer-sub-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun bash -c "for R in 19 22 23 24; do .venv/bin/attn-posteer --group all --every 16 --arms orig --layer \$R --sub attn --alpha 0.25 --max-new-tokens 96 --out work/posteer-sub/L\${R}attn; done \
  && for R in 23 24; do .venv/bin/attn-posteer --group all --every 16 --arms orig --layer \$R --alpha 0.25 --max-new-tokens 96 --out work/posteer-sub/L\$R; done \
  && .venv/bin/python scripts/mergearms.py work/judge-in/sub-a025.jsonl attn19=work/posteer-sub/L19attn/orig.jsonl mlp19=work/posteer-cross/L19/orig.jsonl attn22=work/posteer-sub/L22attn/orig.jsonl mlp22=work/posteer-swap/L22/orig.jsonl attn23=work/posteer-sub/L23attn/orig.jsonl mlp23=work/posteer-sub/L23/orig.jsonl attn24=work/posteer-sub/L24attn/orig.jsonl mlp24=work/posteer-sub/L24/orig.jsonl --alphas=0.25"
