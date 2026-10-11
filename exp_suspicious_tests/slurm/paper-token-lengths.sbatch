#!/bin/bash
#SBATCH --job-name=paper-token-lengths
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/paper-toklen-%j.out
#SBATCH --error=logs/paper-toklen-%j.err
set -euo pipefail
ROOT=${SLURM_SUBMIT_DIR:-$PWD}/../exp_gendata
cd "$ROOT" || exit 1
source .env
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'EOF'
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

rng = np.random.default_rng(0)
table = pq.read_table("data/semantic.parquet", columns=["concept_text", "antagonist_text"])
n = table.num_rows
idx = np.sort(rng.choice(n, size=100_000, replace=False))
texts = []
for col in ("concept_text", "antagonist_text"):
    arr = table.column(col).to_numpy(zero_copy_only=False)
    texts.extend(arr[idx].tolist())
print(f"sampled {len(texts)} stories from {n} rows")

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
lengths = []
for i in range(0, len(texts), 4000):
    enc = tok(texts[i:i + 4000], add_special_tokens=False)
    lengths.extend(len(x) for x in enc["input_ids"])
lengths = np.array(lengths)
words = np.array([len(t.split()) for t in texts])
for name, a in (("tokens", lengths), ("words", words)):
    q5, q50, q95 = np.percentile(a, [5, 50, 95])
    print(f"{name}: q5={q5:.0f} median={q50:.0f} q95={q95:.0f} mean={a.mean():.0f}")
EOF
