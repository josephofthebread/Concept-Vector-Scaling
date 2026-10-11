#!/bin/bash
#SBATCH --job-name=attn-posteer-where
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
#SBATCH --output=logs/posteer-where-%j.out
#SBATCH --error=logs/posteer-where-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
P25=0,43,86,129,172,215,258,301,344,387,430,473,516,559,602,645,688,731,774,817,860,903,946,989,1032
OUT=work/posteer-where
CARRIERS="22:7,22:19,23:3,23:25,27:8,30:0,30:20,30:22,34:15,26:11"
RANDOM_HEADS="22:2,22:12,23:14,23:29,27:21,30:5,30:9,30:27,34:3,26:20"
for L in 19 22 24 28; do
    for A in 0.5 -0.5; do
        srun .venv/bin/attn-posteer --pairs $P25 --arms orig --layer $L --alpha $A --max-new-tokens 96 --out $OUT --where prompt
    done
done
for A in 0.5 -0.5; do
    srun .venv/bin/attn-posteer --pairs $P25 --arms orig --layer 22 --alpha $A --max-new-tokens 96 --out $OUT --where prompt --tag _carriers --ablate "$CARRIERS"
    srun .venv/bin/attn-posteer --pairs $P25 --arms orig --layer 22 --alpha $A --max-new-tokens 96 --out $OUT --where prompt --tag _random --ablate "$RANDOM_HEADS"
done
