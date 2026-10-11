#!/bin/bash
#SBATCH --job-name=attn-vdecomp-lang
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=logs/vdecomp-lang-%j.out
#SBATCH --error=logs/vdecomp-lang-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PAIRS=0,43,86,129,172,215,258,301,344,387,430,473,516,559,602,645,688,731,774,817,860,903,946,989,1032
srun bash -c ".venv/bin/attn-vdecomp --per-pole 8 --language 1 --pairs $PAIRS --out work/vdecomp-de && .venv/bin/attn-vdecomp --per-pole 8 --language 2 --pairs $PAIRS --out work/vdecomp-zh"
