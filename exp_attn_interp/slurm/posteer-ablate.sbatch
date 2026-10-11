#!/bin/bash
#SBATCH --job-name=attn-posteer-ablate
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/posteer-ablate-%j.out
#SBATCH --error=logs/posteer-ablate-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
CARRIERS="22:7,22:19,23:3,23:25,27:8,30:0,30:20,30:22,34:15,26:11"
RANDOM_HEADS="22:2,22:12,23:14,23:29,27:21,30:5,30:9,30:27,34:3,26:20"
srun bash -c "for A in 0.25 0; do \
  .venv/bin/attn-posteer --group all --every 16 --arms orig --layer 22 --alpha \$A --max-new-tokens 96 --out work/posteer-ablate/a\$A --tag _none \
  && .venv/bin/attn-posteer --group all --every 16 --arms orig --layer 22 --alpha \$A --max-new-tokens 96 --out work/posteer-ablate/a\$A --tag _carriers --ablate $CARRIERS \
  && .venv/bin/attn-posteer --group all --every 16 --arms orig --layer 22 --alpha \$A --max-new-tokens 96 --out work/posteer-ablate/a\$A --tag _random --ablate $RANDOM_HEADS; done \
  && .venv/bin/python scripts/mergearms.py work/judge-in/ablate.jsonl none=work/posteer-ablate/a0.25/orig_none.jsonl carriers=work/posteer-ablate/a0.25/orig_carriers.jsonl random=work/posteer-ablate/a0.25/orig_random.jsonl none0=work/posteer-ablate/a0/orig_none.jsonl carriers0=work/posteer-ablate/a0/orig_carriers.jsonl random0=work/posteer-ablate/a0/orig_random.jsonl"
