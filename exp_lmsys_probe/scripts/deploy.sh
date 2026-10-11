#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
remote=${REMOTE:-hse}
root=korznikov_students/dm/exp_lmsys_probe
step=${1:-sync}

rsync -az --delete \
    --exclude="uv.lock" --exclude=".venv" \
    --exclude="work" --exclude="work_smoke" --exclude="logs" \
    --exclude="__pycache__" --exclude="*.pyc" --exclude="*.egg-info" \
    ./ "$remote:$root/"
echo "synced to $remote:$root"
[ "$step" = "sync" ] && exit 0

ssh "$remote" "cd $root && ~/.local/bin/uv sync --python 3.12"

case "$step" in
run)
    ssh "$remote" "cd $root && mkdir -p logs work \
        && rm -f work/cand-*.safetensors work/ids-*.json work/candidates.json work/judged*.json \
        && r=\$(sbatch --parsable slurm/readout.sbatch) \
        && s=\$(sbatch --parsable --dependency=afterok:\$r slurm/select.sbatch) \
        && j=\$(sbatch --parsable --dependency=afterok:\$s slurm/judge.sbatch) \
        && m=\$(sbatch --parsable --dependency=afterok:\$j slurm/merge.sbatch) \
        && echo \"readout=\$r select=\$s judge=\$j merge=\$m\" \
        && squeue -j \$r,\$s,\$j,\$m -o '%.16i %.9P %.13j %.10T %.6M %R'"
    ;;
*)
    echo "unknown step: $step" >&2; exit 1
    ;;
esac
