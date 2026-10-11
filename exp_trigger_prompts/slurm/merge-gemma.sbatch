#!/bin/bash
#SBATCH --job-name=merge-gemma
#SBATCH --partition=cpu-e-quick
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/merge-gemma-%j.out
#SBATCH --error=logs/merge-gemma-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
TAG="google_gemma-2-9b-it"
SHARDS=${SHARDS:?set SHARDS to the steer array width}
for s in $(seq 0 $((SHARDS-1))); do
    f="work/steer/sweep-$TAG-shard-$s.jsonl"
    if [ ! -s "$f" ]; then echo "missing/empty $f -- aborting merge" >&2; exit 1; fi
done
cat work/steer/sweep-$TAG-shard-*.jsonl > "work/steer/sweep-$TAG.jsonl"
echo "=== [$(date +%H:%M:%S)] merged $SHARDS shards, $(wc -l < work/steer/sweep-$TAG.jsonl) rows -> work/steer/sweep-$TAG.jsonl ==="
