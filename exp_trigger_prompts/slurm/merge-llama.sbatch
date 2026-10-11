#!/bin/bash
#SBATCH --job-name=merge-llama
#SBATCH --partition=cpu-e-quick
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/merge-llama-%j.out
#SBATCH --error=logs/merge-llama-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
TAG="unsloth_Meta-Llama-3.1-8B-Instruct"
OUT="work/steer/sweep-$TAG.jsonl"
shards=(work/steer/sweep-"$TAG"-shard-{0..6}.jsonl)
for f in "${shards[@]}"; do
    [ -s "$f" ] || { echo "missing/empty shard: $f" >&2; exit 1; }
done
echo "=== [$(date +%H:%M:%S)] merging ${#shards[@]} shards -> $OUT ==="
cat "${shards[@]}" > "$OUT"
echo "merged rows: $(wc -l < "$OUT")"
for f in "${shards[@]}"; do echo "  $(wc -l < "$f")  $f"; done
