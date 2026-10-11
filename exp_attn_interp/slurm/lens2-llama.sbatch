#!/bin/bash
#SBATCH --job-name=attn-lens2-llama
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:28:00
#SBATCH --output=logs/lens2-llama-%j.out
#SBATCH --error=logs/lens2-llama-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun bash -c 'OMP_NUM_THREADS=8 .venv/bin/attn-gainan --mlpgain work-Meta-Llama-3.1-8B-Instruct/mlpgain --out work-Meta-Llama-3.1-8B-Instruct/gainan --figs figures/gain-llama --band 13,17 && .venv/bin/attn-lens2 --model unsloth/Meta-Llama-3.1-8B-Instruct --vdecomp work-Meta-Llama-3.1-8B-Instruct/vdecomp --markers data/markers_presence-Meta-Llama-3.1-8B-Instruct.json --mlpgain work-Meta-Llama-3.1-8B-Instruct/mlpgain --out work-Meta-Llama-3.1-8B-Instruct/lens2 && .venv/bin/attn-neurlens --model unsloth/Meta-Llama-3.1-8B-Instruct --top work-Meta-Llama-3.1-8B-Instruct/gainan/top_neurons.json --mlpgain work-Meta-Llama-3.1-8B-Instruct/mlpgain --markers data/markers_presence-Meta-Llama-3.1-8B-Instruct.json --out work-Meta-Llama-3.1-8B-Instruct/neurlens'
