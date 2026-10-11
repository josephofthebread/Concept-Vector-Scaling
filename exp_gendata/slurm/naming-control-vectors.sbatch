#!/bin/bash
#SBATCH --job-name=gendata-naming-vectors
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/naming-vectors-%A_%a.out
#SBATCH --error=logs/naming-vectors-%A_%a.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
source .env
MODELS=("Qwen/Qwen3-8B" "google/gemma-2-9b-it" "unsloth/Meta-Llama-3.1-8B-Instruct")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
TAG=$(printf '%s--named_assistant_decl--semantic' "${MODEL//\//_}")
mkdir -p "work/vectors/$TAG" logs
mapfile -t SHARDS < <(find "work/captures/$TAG" -name 'shard-*.safetensors' -type f | sort)
srun .venv/bin/gendata-vectors "${SHARDS[@]}" \
    --ontology "data/semantic.ontology.json" --out "work/vectors/$TAG"
