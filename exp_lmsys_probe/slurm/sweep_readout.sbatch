#!/bin/bash
#SBATCH --job-name=sweep-readout
#SBATCH --partition=rocky
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/sweep-readout-%j.out
#SBATCH --error=logs/sweep-readout-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export PATH="$ROOT/.venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORK="${WORK:-work}"; CORPUS="${CORPUS:-lmsys}"; MODEL="${MODEL:-Qwen/Qwen3-8B}"; mkdir -p logs "$WORK"
echo "=== [$(date +%H:%M:%S)] sweep-readout ${CORPUS} work=$WORK model=$MODEL on $(hostname) ==="
.venv/bin/python -m exp_lmsys_probe.select --work "$WORK" \
    || { echo "select (candidates.json) failed" >&2; exit 1; }
.venv/bin/python -m exp_lmsys_probe.select_examples --work "$WORK" --all --out examples_all.json \
    || { echo "select --all failed" >&2; exit 1; }
.venv/bin/python -m exp_lmsys_probe.hl_readout --work "$WORK" --corpus "$CORPUS" --model "$MODEL" \
    --sel examples_all.json --prefix hla --merged highlights_all.json --compact \
    || { echo "sweep-readout ${CORPUS} failed" >&2; exit 1; }
.venv/bin/python -m exp_lmsys_probe.hl_readout --work "$WORK" --merge --compact \
    --sel examples_all.json --prefix hla --merged highlights_all.json \
    || { echo "sweep-readout merge failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] ${CORPUS} sweep-readout done ==="
