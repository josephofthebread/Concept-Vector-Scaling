#!/bin/bash
#SBATCH --job-name=rh-tokauroc
#SBATCH --partition=cpu-e-quick
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=logs/tokauroc-%j.out
#SBATCH --error=logs/tokauroc-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=${SLURM_SUBMIT_DIR:-$PWD}/.venv-vllm/bin/python
export OMP_NUM_THREADS=8
echo "=== [$(date '+%F %T')] token-distribution AUROC over turnwork_tok/ ==="
$PY -m exp_reward_hack.tokauroc
echo "=== [$(date '+%F %T')] tokauroc done ==="
