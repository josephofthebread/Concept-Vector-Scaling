#!/bin/bash
#SBATCH --job-name=attn-sae-dl
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/sae-dl-%j.out
#SBATCH --error=logs/sae-dl-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
srun .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
import os
layers = [int(x) for x in os.environ.get("LAYERS", "3,11,18,21,23,27,34").split(",")]
p = snapshot_download("Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50", allow_patterns=[f"layer{l}.sae.pt" for l in layers] + ["*.json", "*.md"], max_workers=4)
print("done", p)
PY
