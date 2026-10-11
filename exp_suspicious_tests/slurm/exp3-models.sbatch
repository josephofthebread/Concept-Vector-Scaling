#!/bin/bash
#SBATCH --job-name=suspicious-exp3-models
#SBATCH --partition=rocky
#SBATCH --constraint=type_e|type_f|type_h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/exp3-models-%A-%a.out
#SBATCH --error=logs/exp3-models-%A-%a.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs work/exp3
MODELS=("google/gemma-2-9b-it" "unsloth/Meta-Llama-3.1-8B-Instruct")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SLUG=${MODEL//\//_}
TRANSCRIPTS="work/exp3/transcripts-$SLUG.jsonl"
OUT="work/exp3/proj-$SLUG.safetensors"
if [ -f "$OUT" ]; then echo "exists: $OUT"; exit 0; fi
[ -f "$TRANSCRIPTS" ] || srun .venv/bin/suspicious-generate --model "$MODEL" --out "$TRANSCRIPTS"

.venv/bin/python - "$TRANSCRIPTS" "$MODEL" <<'EOF'
import json, sys
from transformers import AutoTokenizer
path, model = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
row = json.loads(open(path).readline())
print("=== decoded spans, first transcript (hand check) ===")
current, buffer = None, []
for token_id, label in zip(row["input_ids"], row["span_labels"]):
    if label != current:
        if buffer:
            print(f"[{current}] {tok.decode(buffer)!r}"[:180])
        current, buffer = label, []
    buffer.append(token_id)
print(f"[{current}] {tok.decode(buffer)!r}"[:180])
for name in ("emotion_position", "assistant_position", "answer_position"):
    p = row[name]
    piece = tok.decode([row["input_ids"][p]]) if 0 <= p < len(row["input_ids"]) else "?"
    print(f"{name} = {p} -> {piece!r}")
import collections
rows = [json.loads(line) for line in open(path)]
answers = collections.Counter(r["answer"] for r in rows)
print("answers:", dict(answers), "transcripts:", len(rows))
EOF

srun .venv/bin/suspicious-measure --model "$MODEL" --transcripts "$TRANSCRIPTS" --out "$OUT"
