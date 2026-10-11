#!/bin/bash
#SBATCH --job-name=trigger-pipeline
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/pipeline-%j.out
#SBATCH --error=logs/pipeline-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$ROOT/.venv/bin:$PATH"
mkdir -p logs work

SHARDS=2
MODELS=("Qwen/Qwen3-8B" "google/gemma-2-9b-it" "unsloth/Meta-Llama-3.1-8B-Instruct")

wait_all() { local rc=0; for pid in "$@"; do wait "$pid" || rc=1; done; return $rc; }

echo "=== [$(date +%H:%M:%S)] STAGE 1: gpt-oss-120b drafting 1036 x 64 prompts ==="
pids=()
for shard in $(seq 0 $((SHARDS - 1))); do
    echo "--- draft shard $shard on GPU $shard ---"
    CUDA_VISIBLE_DEVICES="$shard" .venv/bin/trigger-generate \
        --out "work/drafts-shard-$shard.jsonl" --shard "$shard" --shards "$SHARDS" \
        > "logs/draft-shard-$shard.log" 2>&1 &
    pids+=("$!")
    sleep 30
done
wait_all "${pids[@]}" || { echo "stage 1 failed; see logs/draft-shard-*.log" >&2; exit 1; }

cat work/drafts-shard-*.jsonl > work/prompts.jsonl
total=$(wc -l < work/prompts.jsonl | tr -d ' ')
usable=$(grep -c '"ok": true' work/prompts.jsonl || true)
echo "=== [$(date +%H:%M:%S)] STAGE 1 done: $total drafts, $usable usable -> work/prompts.jsonl ==="

for model in "${MODELS[@]}"; do
    tag=${model//\//_}
    echo "=== [$(date +%H:%M:%S)] STAGE 2: baselines for $model ==="
    pids=()
    for shard in $(seq 0 $((SHARDS - 1))); do
        echo "--- $tag shard $shard on GPU $shard ---"
        CUDA_VISIBLE_DEVICES="$shard" .venv/bin/trigger-baselines \
            --model "$model" --prompts work/prompts.jsonl \
            --out "work/baselines-$tag-shard-$shard.jsonl" --shard "$shard" --shards "$SHARDS" \
            > "logs/baseline-$tag-shard-$shard.log" 2>&1 &
        pids+=("$!")
        sleep 15
    done
    wait_all "${pids[@]}" || { echo "stage 2 failed for $model; see logs/baseline-$tag-shard-*.log" >&2; exit 1; }
    cat "work/baselines-$tag-shard-"*.jsonl > "work/baselines-$tag.jsonl"
    echo "=== [$(date +%H:%M:%S)] $model done: $(wc -l < "work/baselines-$tag.jsonl" | tr -d ' ') baselines ==="
done

echo "=== [$(date +%H:%M:%S)] pipeline complete ==="
ls -la work/prompts.jsonl work/baselines-*.jsonl
