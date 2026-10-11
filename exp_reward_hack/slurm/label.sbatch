#!/bin/bash
#SBATCH --job-name=rh-label
#SBATCH --partition=cpu-e-quick
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/label-%j.out
#SBATCH --error=logs/label-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=${SLURM_SUBMIT_DIR:-$PWD}/.venv-vllm/bin/python
export OMP_NUM_THREADS=2
echo "=== [$(date '+%F %T')] labeling episodes/ ($(ls episodes/*.json 2>/dev/null | wc -l) episodes) ==="
$PY -m exp_reward_hack.label --dir episodes --out labels.json
echo "=== [$(date '+%F %T')] labeling done -> labels.json ==="
