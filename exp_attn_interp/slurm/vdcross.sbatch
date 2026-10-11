#!/bin/bash
#SBATCH --job-name=attn-vdcross
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:28:00
#SBATCH --output=logs/vdcross-%j.out
#SBATCH --error=logs/vdcross-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun bash -c ".venv/bin/attn-vdcross --vdecomp work/vdecomp --tag qwen --band 19,22 --out work/vdcross.json && .venv/bin/attn-vdcross --vdecomp work-Meta-Llama-3.1-8B-Instruct/vdecomp --tag llama --band 13,17 --out work-Meta-Llama-3.1-8B-Instruct/vdcross.json"
