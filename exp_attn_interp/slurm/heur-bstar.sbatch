#!/bin/bash
#SBATCH --job-name=heur-bstar
#SBATCH --partition=cpu-e-quick
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/heur-bstar-%j.out
#SBATCH --error=logs/heur-bstar-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
srun .venv/bin/python scripts/heur2.py | tee work/tmp/heur-bstar.out
