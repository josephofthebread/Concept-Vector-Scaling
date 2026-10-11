#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
remote=${REMOTE:-hse}
root=korznikov_students/dm/exp_trigger_prompts
model=${MODEL:-openai/gpt-oss-120b}
step=${1:-all}

rsync -az --delete \
    --exclude="uv.lock" --exclude=".venv" \
    --exclude="work" --exclude="logs" --exclude="jobs" \
    --exclude="__pycache__" --exclude="*.pyc" --exclude="*.egg-info" \
    ./ "$remote:$root/"
echo "synced to $remote:$root"
[ "$step" = "sync" ] && exit 0

ssh "$remote" "cd $root && ~/.local/bin/uv sync --python 3.12"

if [ "$step" = "fetch" ]; then
    ssh "$remote" "cd $root && HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_XET=1 .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download('$model',
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'], max_workers=8))
PY"
    exit 0
fi

case "$step" in
    submit)
        ssh "$remote" "cd $root && mkdir -p logs && sbatch slurm/pipeline.sbatch"
        ;;
    probe)
        ssh "$remote" "cd $root && mkdir -p logs && sbatch slurm/probe.sbatch"
        ;;
    steerdemo)
        ssh "$remote" "cd $root && mkdir -p logs && sbatch slurm/steerdemo.sbatch"
        ;;
esac
