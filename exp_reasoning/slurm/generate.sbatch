#!/bin/bash
#SBATCH --job-name=reasoning-generate
#SBATCH --partition=rocky
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
mkdir -p logs work/results
for BENCHMARK in ${BENCHMARKS:-gpqa_diamond math_500}; do
    echo "=== [$(date +%H:%M:%S)] generate $BENCHMARK on $(hostname) ==="
    .venv/bin/python -m exp_reasoning.generate --benchmark "$BENCHMARK" --num-workers 8 --batch-size 4 \
        --max-new-tokens 16384 --results-dir work/results
done
