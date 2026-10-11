#!/bin/bash
#SBATCH --job-name=basis-modes
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/basis-modes-%j.out
#SBATCH --error=logs/basis-modes-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p work/basis
srun .venv/bin/python - <<'EOF'
import json
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

LAYER, K, TOP = 25, 16, 12
with safe_open("../exp_gendata/work/vectors/Qwen_Qwen3-8B--assistant--semantic/diff.safetensors",
               framework="np") as h:
    M = h.get_tensor("diff")[LAYER].astype(np.float64)          # [1036, H]
M /= np.linalg.norm(M, axis=1, keepdims=True)
pairs = json.load(open("../exp_gendata/data/semantic.ontology.json"))["pairs"]
names = [f"{p['concept']} vs {p['antagonist']}" for p in pairs]
classes = [p["class_name"] for p in pairs]

U, s, Vt = np.linalg.svd(M, full_matrices=False)
p = s ** 2 / (s ** 2).sum()
cos = M @ Vt[:K].T                                              # [1036, K] rows unit, modes unit
flip = np.sign(cos[np.abs(cos).argmax(axis=0), np.arange(K)])
cos *= flip
modes = Vt[:K] * flip[:, None]

lines = []
for k in range(K):
    order = np.argsort(cos[:, k])
    lines.append(f"\n=== mode {k:2d}   sigma^2 share {p[k]*100:5.1f}%   "
                 f"cumulative {p[:k+1].sum()*100:5.1f}% ===")
    lines.append("  --- positive pole ---")
    for i in order[::-1][:TOP]:
        lines.append(f"  {cos[i, k]:+.3f}  {names[i][:60]:60s}  [{classes[i][:40]}]")
    lines.append("  --- negative pole ---")
    for i in order[:TOP]:
        lines.append(f"  {cos[i, k]:+.3f}  {names[i][:60]:60s}  [{classes[i][:40]}]")
report = "\n".join(lines)
print(report)
with open("work/basis/modes-qwen-assistant-L25.txt", "w") as f:
    f.write(report + "\n")
save_file({"modes": modes.astype(np.float32), "cos": cos.astype(np.float32),
           "sigma2_share": p[:K].astype(np.float32)},
          "work/basis/modes-qwen-assistant-L25.safetensors")
print("\nwrote work/basis/modes-qwen-assistant-L25.{txt,safetensors}")
EOF
