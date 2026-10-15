#!/bin/bash
#SBATCH --job-name=genvectors-paired-conv
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/paired-conv-%j.out
#SBATCH --error=logs/paired-conv-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
TAGS=(
    Qwen_Qwen3-8B-Base--paired_raw--semantic
    google_gemma-2-9b--paired_raw--semantic
    unsloth_Meta-Llama-3.1-8B--paired_raw--semantic
    Qwen_Qwen3-8B--paired_assistant--semantic
    google_gemma-2-9b-it--paired_assistant--semantic
    unsloth_Meta-Llama-3.1-8B-Instruct--paired_assistant--semantic
    Qwen_Qwen3-8B--named_assistant_decl--semantic
    google_gemma-2-9b-it--named_assistant_decl--semantic
    unsloth_Meta-Llama-3.1-8B-Instruct--named_assistant_decl--semantic
)
for TAG in "${TAGS[@]}"; do
    DEST="work/convergence/$TAG.safetensors"
    [ -s "$DEST" ] && { echo "exists: $DEST"; continue; }
    echo "=== $TAG ==="
    srun .venv/bin/genvectors-convergence --tag "$TAG" --definitions diff
    mv "work/convergence/diff/$TAG.safetensors" "$DEST"
done
echo "all done"
