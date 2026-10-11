#!/bin/bash
#SBATCH --job-name=dose
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export PATH="${SLURM_SUBMIT_DIR:-$PWD}/../exp_lmsys_probe/.venv/bin:$PATH"
mkdir -p results

MODELS=("Qwen/Qwen3-8B" "unsloth/Meta-Llama-3.1-8B-Instruct" "google/gemma-2-9b-it")
TAGS=(qwen llama gemma)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}; T=${TAGS[$SLURM_ARRAY_TASK_ID]}
echo "=== [$(date +%H:%M:%S)] dose readout $M -> results/dose_$T.npz on $(hostname) ==="
timeout -k 30 25m python -m exp_dose.dose_readout --model "$M" --out "results/dose_$T.npz" \
    || { echo "dose readout $T failed (rc=$?)" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] $T done ==="
