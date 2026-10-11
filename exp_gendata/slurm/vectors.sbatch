#!/bin/bash
#SBATCH --job-name=gendata-vectors
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/vectors-%A_%a.out
#SBATCH --error=logs/vectors-%A_%a.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
source .env
read -r MODEL PRESENTATION DATASET < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" jobs/vectors.tsv)
TAG=$(printf '%s--%s--%s' "${MODEL//\//_}" "$PRESENTATION" "$DATASET")
mkdir -p "work/vectors/$TAG" logs
mapfile -t SHARDS < <(find "work/captures/$TAG" -name 'shard-*.safetensors' -type f | sort)
srun .venv/bin/gendata-vectors "${SHARDS[@]}" \
    --ontology "data/$DATASET.ontology.json" --out "work/vectors/$TAG"
