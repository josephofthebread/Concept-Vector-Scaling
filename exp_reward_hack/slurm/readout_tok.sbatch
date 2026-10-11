#!/bin/bash
#SBATCH --job-name=rh-readtok
#SBATCH --partition=rocky
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/readtok-%A_%a.out
#SBATCH --error=logs/readtok-%A_%a.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
VENV=${SLURM_SUBMIT_DIR:-$PWD}/../exp_lmsys_probe/.venv
export PATH="$VENV/bin:$PATH"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

MTAGS=(   qwen  qwen  qwen  llama  llama  llama  gemma  gemma )
KSHARDS=( 3     3     3     3      3      3      2      2     )
SHARD=(   0     1     2     0      1      2      0      1     )
i="$SLURM_ARRAY_TASK_ID"
MTAG="${MTAGS[$i]}"; K="${KSHARDS[$i]}"; S="${SHARD[$i]}"
case "$MTAG" in
    qwen)  MODEL="Qwen/Qwen3-8B" ;;
    llama) MODEL="unsloth/Meta-Llama-3.1-8B-Instruct" ;;
    gemma) MODEL="google/gemma-2-9b-it" ;;
esac

echo "=== [$(date +%H:%M:%S)] readout $MTAG shard $S/$K ($MODEL) on $(hostname) ==="
timeout -k 60 220m python -m exp_reward_hack.turnread_tok --model "$MODEL" \
    --episodes "episodes/*-$MTAG-A-seed*.json" --shard "$S" --shards "$K" --work turnwork_tok \
    || { rc=$?; echo "turnread exited rc=$rc (124=timeout, resumable)"; }
echo "=== [$(date +%H:%M:%S)] done: $(ls turnwork_tok/*-"$MTAG"-A-*.npz 2>/dev/null | wc -l) $MTAG npz on disk ==="
