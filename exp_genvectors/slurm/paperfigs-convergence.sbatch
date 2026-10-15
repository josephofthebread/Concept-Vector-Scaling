#!/bin/bash
#SBATCH --job-name=genvectors-paperfigs
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=logs/paperfigs-%j.out
#SBATCH --error=logs/paperfigs-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MPLBACKEND=Agg
OUT=work/tmp/paperfigs; mkdir -p "$OUT/curves" logs
for t in Qwen3-8B gemma-2-9b-it Meta-Llama-3.1-8B-Instruct; do ln -sfn "$ROOT/work/convcurve-$t" "$OUT/curves/$t"; done
.venv/bin/python scripts/paperfigs/render_convergence_assistant.py --curves "$OUT/curves" --summary work/convcurve \
    --buckets work/convergence --ontology data/semantic.ontology.json --out "$OUT" \
    --numbers "$OUT/paper-numbers.json" --lo 5 --hi 95
ls -la "$OUT"
