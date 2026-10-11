#!/bin/bash
#SBATCH --job-name=attn-posteer-lang-hi
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=logs/posteer-lang-hi-%j.out
#SBATCH --error=logs/posteer-lang-hi-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PAIRS=20,55,56,74,77,100,109,111,161,173,183,193,234,249,280,307,318,369,384,389,396,400,409,424,441,479,489,518,534,558,559,563,576,615,620,629,633,634,659,669,671,681,716,722,796,818,854,856,861,873,949,960,980,1023
ARMS=orig,lang_en,lang_zh,lang_ru,lang_de,langres_en,langres_zh,langst_zh
srun bash -c ".venv/bin/attn-posteer --pairs $PAIRS --arms $ARMS --max-new-tokens 96 --alpha 0.8 --out work/posteer-lang/a80 && .venv/bin/attn-posteer --pairs $PAIRS --arms $ARMS --max-new-tokens 96 --alpha 1.0 --out work/posteer-lang/a100"
