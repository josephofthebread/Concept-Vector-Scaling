#!/bin/bash
#SBATCH --job-name=gendata-convspeed
#SBATCH --partition=cpu-e-quick
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/convspeed-%j.out
#SBATCH --error=logs/convspeed-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs work/convspeed
for TAG in Qwen_Qwen3-8B--assistant--semantic \
           google_gemma-2-9b-it--assistant--semantic \
           unsloth_Meta-Llama-3.1-8B-Instruct--assistant--semantic; do
    .venv/bin/python scripts/convspeed.py --tag "$TAG" || { echo "convspeed failed for $TAG" >&2; exit 1; }
done
echo "=== [$(date +%H:%M:%S)] convspeed done ==="
