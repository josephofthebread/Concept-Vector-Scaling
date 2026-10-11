#!/bin/bash
#SBATCH --job-name=suspicious-report
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/report-%j.out
#SBATCH --error=logs/report-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs work/figures
for EXPERIMENT in ${EXPERIMENTS:-exp1 exp3}; do
    case "$EXPERIMENT" in
    exp1|exp3) DATA=work/exp1 ;;
    exp2) DATA=work/exp2 ;;
    esac
    srun .venv/bin/suspicious-report "$EXPERIMENT" --data "$DATA" --figures work/figures
done
