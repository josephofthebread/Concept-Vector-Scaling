#!/bin/bash
#SBATCH --job-name=attn-posteer-kind
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=logs/posteer-kind-%j.out
#SBATCH --error=logs/posteer-kind-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
P25=0,43,86,129,172,215,258,301,344,387,430,473,516,559,602,645,688,731,774,817,860,903,946,989,1032
OUT=work/posteer-kind
for A in 0.5 -0.5; do
    srun .venv/bin/attn-posteer --pairs $P25 --arms ugap,egap,lex1 --layer 22 --alpha $A --max-new-tokens 96 --out $OUT
done
srun .venv/bin/python scripts/footprint.py --records "$OUT/ugap.jsonl;$OUT/egap.jsonl;$OUT/lex1.jsonl;work/posteer-foot/orig.jsonl" \
    --labels work/sae/label-instruct.json --rows 22,24,28 --out work/footprint/kind.jsonl
