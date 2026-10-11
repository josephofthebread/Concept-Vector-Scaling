#!/bin/bash
#SBATCH --job-name=gendata-lexfeat
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/lexfeat-%j.out
#SBATCH --error=logs/lexfeat-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'EOF'
import re, json
import numpy as np
import pyarrow.parquet as pq
from collections import Counter, defaultdict

WORD = re.compile(r"[\w']+")
table = pq.read_table("data/semantic.parquet",
                      columns=["pair_id", "concept", "antagonist", "concept_text", "antagonist_text"]).to_pydict()
vocab = defaultdict(lambda: (Counter(), Counter()))
lengths = defaultdict(lambda: [0.0, 0.0, 0])
hits = defaultdict(lambda: [0, 0, 0])
for pid, c, a, ct, at in zip(*(table[k] for k in ("pair_id", "concept", "antagonist", "concept_text", "antagonist_text"))):
    wc, wa = vocab[pid]
    tc = [w.lower() for w in WORD.findall(ct)]
    ta = [w.lower() for w in WORD.findall(at)]
    wc.update(tc); wa.update(ta)
    L = lengths[pid]; L[0] += len(tc); L[1] += len(ta); L[2] += 1
    H = hits[pid]; H[0] += c.casefold() in ct.casefold(); H[1] += a.casefold() in at.casefold(); H[2] += 1
pairs = sorted(vocab)
out = {"pair_id": [], "vocab_jaccard": [], "unigram_cosine": [], "label_rate": [], "mean_tokens": [], "length_ratio": []}
for pid in pairs:
    wc, wa = vocab[pid]
    sc, sa = set(wc), set(wa)
    jac = len(sc & sa) / len(sc | sa)
    common = sc & sa
    num = sum(wc[w] * wa[w] for w in common)
    cos = num / (np.sqrt(sum(v*v for v in wc.values())) * np.sqrt(sum(v*v for v in wa.values())))
    L, H = lengths[pid], hits[pid]
    out["pair_id"].append(pid)
    out["vocab_jaccard"].append(jac)
    out["unigram_cosine"].append(float(cos))
    out["label_rate"].append((H[0] + H[1]) / (2 * H[2]))
    out["mean_tokens"].append((L[0] + L[1]) / (2 * L[2]))
    out["length_ratio"].append(L[0] / max(L[1], 1))
import pyarrow as pa
pq.write_table(pa.table(out), "work/convergence/lexical_features.parquet")
print("pairs:", len(pairs))
EOF
