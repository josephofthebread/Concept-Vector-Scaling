#! /usr/bin/env bash
set -u

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
GPUS=${GPUS:-"0 1 2 3"}
QUEUE=${QUEUE:?set QUEUE to a job file}
OUT=${OUT:-episodes/impossible}
DTYPE=${DTYPE:-float16}
LOCK="$QUEUE.lock"

mkdir -p "$OUT"
touch "$LOCK"

for gpu in $GPUS; do
    (
        while true; do
            job=$(flock "$LOCK" bash -c "head -n 1 '$QUEUE'; sed -i '1d' '$QUEUE'")
            [ -z "$job" ] && break
            # shellcheck disable=SC2086
            set -- $job
            split=$1; task=$2; seed=$3
            echo "$(date +%T) gpu$gpu start ${split}/${task} seed${seed}"
            CUDA_VISIBLE_DEVICES=$gpu IMPOSSIBLE_SPLIT=$split IMPOSSIBLE_TASK=$task \
                .venv/bin/python -m exp_agentic.agent --workload 10_impossiblebench --variant base \
                    --seed "$seed" --dtype "$DTYPE" --out "$OUT" \
                    > "$OUT/${split}-${task}-seed${seed}.log" 2>&1
            echo "$(date +%T) gpu$gpu done  ${split}/${task} seed${seed} rc=$?"
        done
        echo "$(date +%T) gpu$gpu drained"
    ) &
done
wait

docker ps -q | xargs -r docker rm -f > /dev/null
echo "$(date +%T) complete: $(ls "$OUT"/*.json 2>/dev/null | wc -l) episodes in $OUT"
