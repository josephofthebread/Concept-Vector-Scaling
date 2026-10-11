#!/bin/bash
#SBATCH --job-name=lmsys-merge
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/merge-%j.out
#SBATCH --error=logs/merge-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export PATH="$ROOT/.venv/bin:$PATH"
WORK="${WORK:-work}"
echo "=== [$(date +%H:%M:%S)] judge-merge work=$WORK ==="
.venv/bin/python -m exp_lmsys_probe.judge --work "$WORK" --merge || { echo "merge failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] done: $(ls -la "$WORK"/judged.json 2>/dev/null) ==="
