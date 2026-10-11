#!/bin/bash
#SBATCH --job-name=gendata-sentdata
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/sentdata-%j.out
#SBATCH --error=logs/sentdata-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
source .env
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'EOF'
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer

CATEGORIES = ["Books", "Electronics", "Home_and_Kitchen", "Clothing_Shoes_and_Jewelry",
              "Beauty_and_Personal_Care", "Toys_and_Games", "Sports_and_Outdoors",
              "Automotive", "Pet_Supplies", "Grocery_and_Gourmet_Food"]
STARS = (1, 2, 4, 5)
PER_CELL = 100
BAND = (308, 338)  # measured story-length [q45, q55] band, Qwen tokens
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True, use_fast=True)
rows = []
for cat in CATEGORIES:
    got = {s: 0 for s in STARS}
    stream = load_dataset("json",
                          data_files=f"hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/{cat}.jsonl",
                          split="train", streaming=True)
    scanned = 0
    for r in stream:
        scanned += 1
        stars = int(r["rating"])
        if stars not in got or got[stars] >= PER_CELL:
            if all(v >= PER_CELL for v in got.values()):
                break
            continue
        text = (r.get("title") or "") + "\n" + (r.get("text") or "")
        n = len(tok(text, add_special_tokens=False, truncation=True, max_length=1024)["input_ids"])
        if BAND[0] <= n <= BAND[1]:
            rows.append({"category": cat, "stars": stars, "text": text, "tokens": n})
            got[stars] += 1
        if scanned % 200000 == 0:
            print(cat, scanned, got, flush=True)
    print("DONE", cat, "scanned", scanned, got, flush=True)
pq.write_table(pa.Table.from_pylist(rows), "data/sentiment.parquet", compression="zstd")
print("total rows:", len(rows))
EOF
