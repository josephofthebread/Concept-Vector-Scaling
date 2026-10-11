#!/bin/bash
#SBATCH --job-name=lmsys-figures
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/figures-%j.out
#SBATCH --error=logs/figures-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export PATH="$ROOT/.venv/bin:$PATH"
WORK="${WORK:-work}"
echo "=== [$(date +%H:%M:%S)] figures work=$WORK on $(hostname) ==="
.venv/bin/python -m exp_lmsys_probe.outcome --work "$WORK" || { echo "outcome failed" >&2; exit 1; }
.venv/bin/python -m exp_lmsys_probe.render --work "$WORK" --figures "$WORK/figures" \
    || { echo "render failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] done: $(ls "$WORK"/figures/*.svg 2>/dev/null | wc -l) SVGs ==="
