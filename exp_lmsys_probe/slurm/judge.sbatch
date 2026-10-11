#!/bin/bash
#SBATCH --job-name=lmsys-judge
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/judge-%A_%a.out
#SBATCH --error=logs/judge-%A_%a.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH="$ROOT/.venv/bin:$PATH"
WORK="${WORK:-work}"; mkdir -p logs "$WORK"
echo "=== [$(date +%H:%M:%S)] judge shard ${SLURM_ARRAY_TASK_ID}/6 work=$WORK on $(hostname) ==="
.venv/bin/python -m exp_lmsys_probe.judge --work "$WORK" --corpus "${JCORPUS:-lmsys}" \
    --shard "${SLURM_ARRAY_TASK_ID}" --shards 6 \
    ${MAXLEN:+--max-model-len "$MAXLEN"} ${GPUMEM:+--gpu-memory-utilization "$GPUMEM"} \
    || { echo "judge shard ${SLURM_ARRAY_TASK_ID} failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] shard ${SLURM_ARRAY_TASK_ID} done ==="
