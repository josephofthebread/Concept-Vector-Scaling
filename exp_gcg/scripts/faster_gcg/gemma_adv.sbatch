#!/bin/bash
#SBATCH --job-name=gemma2it-adv
#SBATCH --account=proj_1887
#SBATCH --partition=rocky
#SBATCH --constraint=type_e
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/adv-%A-%a.log
#SBATCH --error=logs/adv-%A-%a.err
set -euo pipefail
cd "${FASTER_GCG:?set FASTER_GCG to a Faster-GCG checkout with faster-gcg.patch applied}" || exit 1
module purge
module load Python/PyTorch_GPU_v2.4
set -a; source .env; set +a
export HF_HOME="$PWD/.hf_cache"
export HF_DATASETS_CACHE="$PWD/.hf_cache/datasets"
.venv/bin/python -u tools/main.py -c configs/default.yaml -c configs/models/gemma2_9b_it.yaml -c configs/datasets/advbench.yaml --workers 1 --overwrite --dataset-split train --subset 0 --dataset-shard-index "$SLURM_ARRAY_TASK_ID" --dataset-shard-count 4 --expr-path "output/gemma2_9b_it/advbench/shard-$SLURM_ARRAY_TASK_ID"
