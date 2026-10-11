#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
remote=${REMOTE:-hse}
root=korznikov_students/dm/exp_attn_interp
step=${1:-all}

rsync -az --delete \
    --exclude="uv.lock" --exclude=".venv" \
    --exclude="work" --exclude="work-*" --exclude="figures" --exclude="logs" --exclude="jobs" \
    --exclude="data/markers_presence.json" --exclude="data/markers_logitlens.json" \
    --exclude="data/markers_*" --exclude="deshare_cmp.py" \
    --exclude="__pycache__" --exclude="*.pyc" --exclude="*.egg-info" \
    ./ "$remote:$root/"
echo "synced to $remote:$root"
[ "$step" = "sync" ] && exit 0

ssh "$remote" "cd $root && ~/.local/bin/uv sync --python 3.12"
