#! /usr/bin/env bash
set -u

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
GPUS=${GPUS:-"0 1 2 3"}
QUEUE=${QUEUE:?set QUEUE to a file of "<variant> <alpha> <seed>" lines}
OUT=${OUT:-episodes/steer25}
PAIR=${PAIR:-872}
NORM_FROM=${NORM_FROM:-episodes/fixed/impossible_tests-judge-seed3.json}
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
            variant=$1; alpha=$2; seed=$3
            echo "$(date +%T) gpu$gpu start $variant a$alpha seed$seed"
            CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -m exp_agentic.agent \
                --workload impossible_tests --variant "$variant" --seed "$seed" \
                --concept "$PAIR" --alpha "$alpha" --steer-layer 25 --norm-from "$NORM_FROM" \
                --out "$OUT" > "$OUT/$variant-a$alpha-seed$seed.log" 2>&1
            echo "$(date +%T) gpu$gpu done  $variant a$alpha seed$seed rc=$?"
        done
        echo "$(date +%T) gpu$gpu drained"
    ) &
done
wait
echo "$(date +%T) dispatch complete: $(ls "$OUT"/*.json 2>/dev/null | wc -l) episodes in $OUT"
