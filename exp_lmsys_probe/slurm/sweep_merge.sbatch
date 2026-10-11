#!/bin/bash
#SBATCH --job-name=sweep-merge
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=logs/sweep-merge-%j.out
#SBATCH --error=logs/sweep-merge-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export PATH="$ROOT/.venv/bin:$PATH"
OUTDIR="${OUTDIR:-work/span_sweep}"
echo "=== [$(date +%H:%M:%S)] sweep-merge out=$OUTDIR on $(hostname) ==="
.venv/bin/python -m exp_lmsys_probe.span_judge --sweep-merge --out "$OUTDIR" \
    || { echo "sweep-merge $OUTDIR failed" >&2; exit 1; }
echo "=== [$(date +%H:%M:%S)] sweep-merge done ==="
