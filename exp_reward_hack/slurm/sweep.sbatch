#!/bin/bash
#SBATCH --job-name=rh-sweep
#SBATCH --partition=rocky
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/sweep-%A_%a.out
#SBATCH --error=logs/sweep-%A_%a.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
VENV=${SLURM_SUBMIT_DIR:-$PWD}/.venv-vllm
export PATH="$VENV/bin:$PATH"
export AGENTIC_SANDBOX=subprocess
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING
export OMP_NUM_THREADS=8

MTAGS=(   qwen  qwen  qwen  llama  llama  llama  gemma  gemma )
STARTS=(  0     1     2     0      1      2      0      1     )
STRIDES=( 3     3     3     3      3      3      2      2     )
NSEEDS=(  60    60    60    60     60     60     90     90    )
i="$SLURM_ARRAY_TASK_ID"
MTAG="${MTAGS[$i]}"; START="${STARTS[$i]}"; STRIDE="${STRIDES[$i]}"; SEEDS="${NSEEDS[$i]}"
case "$MTAG" in
    qwen)  MODEL="Qwen/Qwen3-8B" ;;
    llama) MODEL="unsloth/Meta-Llama-3.1-8B-Instruct" ;;
    gemma) MODEL="google/gemma-2-9b-it" ;;
    *) echo "unknown model tag '$MTAG'" >&2; exit 2 ;;
esac

echo "=== [$(date +%H:%M:%S)] sweep task $i: $MTAG start=$START stride=$STRIDE seeds=$SEEDS ($MODEL) on $(hostname) ==="
timeout -k 60 700m python -m exp_reward_hack.runmany --model "$MODEL" --backend vllm --config A \
    --seed-start "$START" --seed-stride "$STRIDE" --seeds "$SEEDS" --out episodes \
    || { rc=$?; echo "runmany exited rc=$rc (124=timeout, resumable)"; }
echo "=== [$(date +%H:%M:%S)] done: $(ls episodes/*-"$MTAG"-A-*.json 2>/dev/null | wc -l) $MTAG episodes on disk ==="
