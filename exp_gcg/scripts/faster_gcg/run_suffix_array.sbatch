#!/bin/bash
#SBATCH --job-name=faster-gcg
#SBATCH --account=proj_1887
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-11%12
#SBATCH --output=logs/faster-gcg-%A-%a.log
#SBATCH --error=logs/faster-gcg-%A-%a.err

set -euo pipefail
cd "${FASTER_GCG:?set FASTER_GCG to a Faster-GCG checkout with faster-gcg.patch applied}" || exit 1
module purge
module load Python/PyTorch_GPU_v2.4
source .env

models=(qwen3_8b gemma2_9b llama3_1_8b_instruct)
datasets=(advbench jbb)
model=${models[$((SLURM_ARRAY_TASK_ID / 4))]}
dataset=${datasets[$(((SLURM_ARRAY_TASK_ID % 4) / 2))]}
shard=$((SLURM_ARRAY_TASK_ID % 2))

.venv/bin/python -u tools/main.py \
  -c configs/default.yaml \
  -c "configs/models/${model}.yaml" \
  -c "configs/datasets/${dataset}.yaml" \
  --workers 1 \
  --overwrite \
  --dataset-shard-index "${shard}" \
  --dataset-shard-count 2 \
  --expr-path "output/${model}/${dataset}/shard-${shard}"
