#!/bin/bash
#SBATCH --job-name=turnread
#SBATCH --partition=rocky
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/turn-%j.out
#SBATCH --error=logs/turn-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
PY=${SLURM_SUBMIT_DIR:-$PWD}/../exp_lmsys_probe/.venv/bin/python
WORK="${WORK:-turnwork}"
mkdir -p logs "$WORK"
echo "=== [$(date +%H:%M:%S)] turnread on $(hostname); episodes: $(ls episodes/gate episodes/ib 2>/dev/null | wc -l) ==="

if [ -n "$(ls "$WORK"/*.npz 2>/dev/null)" ]; then echo "skip readout ($WORK has npz)"; else
    timeout -k 120 2h "$PY" -m exp_agentic.turnread --episodes episodes/gate episodes/ib --work "$WORK" \
        || { echo "turnread failed (rc=$?)" >&2; exit 1; }
fi
OMP_NUM_THREADS=8 "$PY" -m exp_agentic.turndose --work "$WORK" --out "$WORK/turndose.json" \
    || { echo "turndose failed (rc=$?)" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] done: $(ls -la "$WORK"/turndose.json 2>/dev/null) ==="
