#!/bin/bash
#SBATCH --job-name=faster-gcg-advleft
#SBATCH --account=proj_1887
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-7%8
#SBATCH --output=logs/faster-gcg-advleft-%A-%a.log
#SBATCH --error=logs/faster-gcg-advleft-%A-%a.err

set -euo pipefail
cd "${FASTER_GCG:?set FASTER_GCG to a Faster-GCG checkout with faster-gcg.patch applied}" || exit 1
module purge
module load Python/PyTorch_GPU_v2.4
set -a
source .env
set +a
export HF_DATASETS_CACHE="$PWD/.hf_cache/datasets"

models=(qwen3_8b qwen3_8b qwen3_8b gemma2_9b gemma2_9b gemma2_9b llama3_1_8b_instruct llama3_1_8b_instruct)
shards=(0 1 2 0 1 2 0 1)
shard_counts=(3 3 3 3 3 3 2 2)
model=${models[$SLURM_ARRAY_TASK_ID]}
shard=${shards[$SLURM_ARRAY_TASK_ID]}
shard_count=${shard_counts[$SLURM_ARRAY_TASK_ID]}

.venv/bin/python -u tools/main.py \
  -c configs/default.yaml \
  -c "configs/models/${model}.yaml" \
  -c configs/datasets/advbench.yaml \
  --workers 1 \
  --overwrite \
  --dataset-split train \
  --subset 0 \
  --dataset-skip 100 \
  --dataset-shard-index "${shard}" \
  --dataset-shard-count "${shard_count}" \
  --expr-path "output/${model}/advbench_remaining/shard-${shard}"
