#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
remote=${REMOTE:-hse}
root=korznikov_students/dm/exp_suspicious_tests
step=${1:-all}

rsync -az --delete \
    --exclude="uv.lock" --exclude=".venv" \
    --exclude="work" --exclude="logs" --exclude="jobs" \
    --exclude="data/mcqa.parquet" --exclude="data/lmsys_items.parquet" --exclude="data/prefills-*.json" \
    --exclude="__pycache__" --exclude="*.pyc" --exclude="*.egg-info" \
    ./ "$remote:$root/"
echo "synced to $remote:$root"
[ "$step" = "sync" ] && exit 0

ssh "$remote" "cd $root && ~/.local/bin/uv sync --python 3.12"
case "$step" in
pilot05)
    ssh "$remote" "cd $root && mkdir -p logs && sbatch slurm/pilot05.sbatch"
    ;;
pilot15)
    ssh "$remote" "cd $root && mkdir -p logs && sbatch slurm/pilot15.sbatch"
    ;;
esac
