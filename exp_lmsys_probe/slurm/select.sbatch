#!/bin/bash
#SBATCH --job-name=lmsys-select
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=logs/select-%j.out
#SBATCH --error=logs/select-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=8 PATH="$ROOT/.venv/bin:$PATH"
WORK="${WORK:-work}"
echo "=== [$(date +%H:%M:%S)] select work=$WORK ==="
.venv/bin/python -m exp_lmsys_probe.select --work "$WORK" || { echo "select failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] done: $(ls -la "$WORK"/candidates.json 2>/dev/null) ==="
