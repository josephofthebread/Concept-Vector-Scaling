#!/bin/bash
#SBATCH --job-name=attn-ablate
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/ablate-%j.out
#SBATCH --error=logs/ablate-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs work
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun .venv/bin/attn-decompan --band-start "${BAND:-19}" --edgemap work/stories_rw/edgemap.json
srun .venv/bin/attn-ablate --model "${MODEL:-Qwen/Qwen3-8B}" --from-decomp "${DECOMP:-work/per_concept_decomp.json}" \
    --n-pairs "${NPAIRS:-48}" --top "${TOP:-5}" --controls "${CONTROLS:-5}" --per-pole "${PER_POLE:-4}" \
    --out "${OUT:-work/ablate.json}"
