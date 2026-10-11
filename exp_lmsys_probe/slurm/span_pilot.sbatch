#!/bin/bash
#SBATCH --job-name=span-pilot
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/span-pilot-%j.out
#SBATCH --error=logs/span-pilot-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export VLLM_USE_FLASHINFER_SAMPLER=0 PATH="$ROOT/.venv/bin:$PATH"
EFFORTS="${EFFORTS:-low}"; OUTDIR="${OUTDIR:-work/span_pilot}"; DESCMODE="${DESCMODE:-full}"
mkdir -p logs "$OUTDIR"
echo "=== [$(date +%H:%M:%S)] span-pilot efforts=$EFFORTS desc-mode=$DESCMODE out=$OUTDIR on $(hostname) ==="
.venv/bin/python -m exp_lmsys_probe.span_judge --efforts "$EFFORTS" --out "$OUTDIR" --desc-mode "$DESCMODE" \
    ${MAXLEN:+--max-model-len "$MAXLEN"} ${GPUMEM:+--gpu-memory-utilization "$GPUMEM"} \
    || { echo "span-pilot failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] span-pilot done ==="
