#!/bin/bash
#SBATCH --job-name=gendata-decl-conv-fix
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=logs/decl-conv-fix-%j.out
#SBATCH --error=logs/decl-conv-fix-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'PY'
import glob, json
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
from exp_gendata.convergence import families, subset_vectors, cosine_rows, SUBSETS

TAGS = ["Qwen_Qwen3-8B--named_assistant_decl--semantic",
        "google_gemma-2-9b-it--named_assistant_decl--semantic",
        "unsloth_Meta-Llama-3.1-8B-Instruct--named_assistant_decl--semantic"]
spec = families()
for tag in TAGS:
    files = sorted(glob.glob(f"work/captures/{tag}/shard-*.safetensors"))
    assert len(files) == 4, f"{tag}: {len(files)} shards"
    sums = counts = None
    for f in files:
        with safe_open(f, framework="np") as h:
            c = h.get_tensor("counts").astype(np.float64)   # [pair,side,bucket]
            s = h.get_tensor("sums")
            s = (s.astype(np.float64) * np.maximum(c, 1)[None, ..., None]) if s.dtype == np.float16 else s.astype(np.float64)
        sums = s if sums is None else sums + s
        counts = c if counts is None else counts + c
    layers, pairs = sums.shape[0], sums.shape[1]
    scores = {name: np.empty((layers, len(combos), pairs), dtype=np.float32) for name, combos in spec.items()}
    for layer in range(layers):
        base = {sub: subset_vectors(sums[layer], counts, sub)["diff"] for sub in SUBSETS}
        for name, combos in spec.items():
            for pos, (a, b) in enumerate(combos):
                scores[name][layer, pos] = cosine_rows(base[a], base[b])
    meta = {"manifest": json.dumps({"tag": tag, "vector": "diff", "shards_merged": len(files),
            "axes": ["layer", "combo", "pair"],
            "combos": {n: [[list(a), list(b)] for a, b in c] for n, c in spec.items()}}, sort_keys=True)}
    save_file(scores, f"work/convergence/{tag}.safetensors", metadata=meta)
    print(f"{tag}: rewrote convergence on {len(files)} merged shards", flush=True)
print("done")
PY
