#!/bin/bash
#SBATCH --job-name=basis-radar
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/basis-radar-%j.out
#SBATCH --error=logs/basis-radar-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'EOF'
import numpy as np
from safetensors import safe_open
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K = 6
NAMES = ["valence", "order &\nboundaries", "authenticity", "self-\nexpansion",
         "epistemic\nhumility", "traditionalism"]
with safe_open("work/basis/modes-qwen-assistant-L25.safetensors", framework="np") as h:
    cos = h.get_tensor("cos")[:, :K].astype(np.float64)        # [1036, K]
    share = h.get_tensor("sigma2_share")[:K]
E = cos ** 2                                                    # energy fraction per mode
ang = np.linspace(0, 2 * np.pi, K, endpoint=False)
ang_c = np.concatenate([ang, ang[:1]])

fig, ax = plt.subplots(figsize=(5.2, 5.2), subplot_kw={"projection": "polar"})
for row in E:
    ax.plot(ang_c, np.concatenate([row, row[:1]]), color="#777", alpha=0.03, linewidth=0.5)
med = np.median(E, axis=0)
p90 = np.percentile(E, 90, axis=0)
ax.plot(ang_c, np.concatenate([med, med[:1]]), color="#c62828", linewidth=1.8)
ax.plot(ang_c, np.concatenate([p90, p90[:1]]), color="#1565c0", linewidth=1.4, linestyle="--")
ax.set_xticks(ang)
ax.set_xticklabels([f"{n}\n({s*100:.0f}%)" for n, s in zip(NAMES, share)], fontsize=8)
ax.set_rlabel_position(90)
ax.tick_params(axis="y", labelsize=7)
ax.set_ylim(0, min(1.0, float(E.max()) * 1.05))
fig.tight_layout()
for out in ("work/basis/radar-qwen-assistant-L25.svg", "work/basis/radar-qwen-assistant-L25.pdf"):
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
print("median energy per mode:", np.round(med, 3))
print("median total energy in 6 modes:", round(float(np.median(E.sum(1))), 3))
EOF
