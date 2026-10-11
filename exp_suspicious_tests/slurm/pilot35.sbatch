#!/bin/bash
#SBATCH --job-name=suspicious-pilot35
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/pilot35-%j.out
#SBATCH --error=logs/pilot35-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs work/exp4
srun .venv/bin/suspicious-exp4 --chunk 0

.venv/bin/python - <<'EOF'
import json
from pathlib import Path
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)
paths = sorted(Path("work/exp4/transcripts").glob("item-*.json"))
print(f"=== {len(paths)} transcripts ===")
row = json.loads(paths[0].read_text())
print("first item text:", row["text"][:150].replace("\n", " | "))
print("reply:", row["reply"][:150].replace("\n", " | "))
print("sinks:", row["sink_tokens"], "tokens:", len(row["input_ids"]))
for entry in row["positions"]:
    if entry["kind"] != "grid":
        piece = tok.decode([row["input_ids"][entry["position"]]])
        print(f"  {entry['kind']:14s} @ {entry['position']:4d} -> {piece!r}")
grid = sum(1 for e in row["positions"] if e["kind"] == "grid")
print("grid positions:", grid)
sizes = [p.stat().st_size for p in Path("work/exp4/rho").glob("item-*.safetensors")]
print(f"rho files: {len(sizes)}, mean {sum(sizes)/len(sizes)/2**20:.1f} MiB, "
      f"projected sweep total {sum(sizes)/len(sizes)*1024/2**30:.1f} GiB")
EOF
