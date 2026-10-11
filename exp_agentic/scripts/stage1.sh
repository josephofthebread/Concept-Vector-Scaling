#! /usr/bin/env bash
set -u

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
OUT=${OUT:-episodes/stage1}
mkdir -p "$OUT"

docker ps -aq --filter ancestor=neuralprobes-agentic:1 | xargs -r docker rm -f >/dev/null

for gpu in 0 1 2 3; do
    (
        for index in $(seq 0 15); do
            [ $((index % 4)) -eq "$gpu" ] || continue
            if [ $((index / 8)) -eq 0 ]; then variant=readme; else variant=judge; fi
            seed=$((index % 8))
            echo "$(date +%T) gpu$gpu start $variant seed$seed"
            CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -m exp_agentic.agent \
                --workload impossible_tests --variant "$variant" --seed "$seed" --out "$OUT" \
                > "$OUT/$variant-seed$seed.log" 2>&1
            echo "$(date +%T) gpu$gpu done  $variant seed$seed rc=$?"
        done
    ) &
done
wait

docker ps -aq --filter ancestor=neuralprobes-agentic:1 | xargs -r docker rm -f >/dev/null
echo "$(date +%T) stage 1 complete: $(ls "$OUT"/*.json 2>/dev/null | wc -l) episodes"
