#!/bin/bash
#SBATCH --job-name=rh-hist
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=logs/hist-%j.out
#SBATCH --error=logs/hist-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
V=${SLURM_SUBMIT_DIR:-$PWD}/../exp_lmsys_probe/.venv
export PATH="$V/bin:$PATH"
export OMP_NUM_THREADS=8 MPLBACKEND=Agg
echo "=== [$(date '+%T')] hist_auroc on $(hostname) ==="
python -m exp_reward_hack.hist_auroc
echo "=== [$(date '+%T')] done ==="
