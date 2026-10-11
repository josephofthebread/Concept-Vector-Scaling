#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
remote=${REMOTE:-hse}
root=korznikov_students/dm/exp_gendata
step=${1:-all}

rsync -az --delete \
    --exclude=".env" --exclude="uv.lock" --exclude=".venv" --exclude=".venv-fa" --exclude=".venv-v100" \
    --exclude="data" --exclude="work" --exclude="logs" --exclude="jobs" \
    --exclude="__pycache__" --exclude="*.pyc" --exclude=".pytest_cache" --exclude="*.egg-info" \
    ./ "$remote:$root/"
echo "synced to $remote:$root"
[ "$step" = "sync" ] && exit 0

ssh "$remote" "cd $root && ~/.local/bin/uv sync --python 3.12"
