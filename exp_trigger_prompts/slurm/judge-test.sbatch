#!/bin/bash
#SBATCH --job-name=judge-test
#SBATCH --partition=rocky
#SBATCH --constraint=type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:25:00
#SBATCH --output=logs/judge-test-%j.out
#SBATCH --error=logs/judge-test-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$ROOT/.venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs work/judge-test
echo "=== [$(date +%H:%M:%S)] judge test on $(nvidia-smi -L | head -1) ==="
srun .venv/bin/trigger-judge \
    --steered work/steer/sweep-Qwen_Qwen3-8B.jsonl --features data/features.json \
    --out-dir work/judge-test --shard 0 --shards 8 --prompts 8 --limit 60
echo "=== parse rate (concept) ==="
.venv/bin/python - <<'PY'
import json, collections
c=collections.Counter(json.loads(l)["score"] for l in open("work/judge-test/concept-shard-0.jsonl"))
print("concept dist:", dict(sorted(c.items())))
PY
