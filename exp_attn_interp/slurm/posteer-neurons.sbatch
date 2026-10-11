#!/bin/bash
#SBATCH --job-name=attn-posteer-neurons
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:35:00
#SBATCH --output=logs/posteer-neurons-%j.out
#SBATCH --error=logs/posteer-neurons-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
P25=0,43,86,129,172,215,258,301,344,387,430,473,516,559,602,645,688,731,774,817,860,903,946,989,1032
OUT=work/posteer-neurons
for A in 0.5 -0.5 0; do
    srun .venv/bin/attn-posteer --pairs $P25 --arms orig --layer 22 --alpha $A --max-new-tokens 96 --out $OUT --tag _keyed --neurons work/tmp/keyed.json:keyed
    srun .venv/bin/attn-posteer --pairs $P25 --arms orig --layer 22 --alpha $A --max-new-tokens 96 --out $OUT --tag _rndn --neurons work/tmp/keyed.json:random
done
srun .venv/bin/python scripts/mergearms.py work/judge-in/neurons.jsonl keyed=$OUT/orig_keyed.jsonl rndn=$OUT/orig_rndn.jsonl
