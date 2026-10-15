#!/bin/bash
#SBATCH --job-name=suspicious-exp3-qwenplot
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/exp3-qwenplot-%j.out
#SBATCH --error=logs/exp3-qwenplot-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs work/figures
srun .venv/bin/python - <<'EOF'
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from safetensors import safe_open

pairs = json.loads(Path("../exp_genvectors/data/semantic.ontology.json").read_text())["pairs"]
pairs.sort(key=lambda pair: int(pair["pair_id"]))
SNR_MIN = 10.0


def load(arm):
    with safe_open("work/exp3/proj-Qwen_Qwen3-8B-deepseek.safetensors", framework="np") as handle:
        proj = np.nan_to_num(handle.get_tensor("proj").astype(np.float64))
        meta = json.loads(handle.metadata()["meta"])
        positions = json.loads(handle.metadata()["positions"])
    pi = positions.index("answer")
    levels = np.array([m["level"] for m in meta])
    arms = np.array([m["arm"] for m in meta])
    keep = arms == arm
    values, lv = proj[keep, pi], levels[keep]
    means = np.stack([values[lv == level].mean(axis=0) for level in range(6)])
    sems = np.stack([values[lv == level].std(axis=0, ddof=1) / np.sqrt((lv == level).sum())
                     for level in range(6)])
    return means, sems


means, sems = load("main")
rmeans, rsems = load("roleplay")
steps = np.diff(means, axis=0)
mono = (steps > 0).all(axis=0) | (steps < 0).all(axis=0)
snr = np.abs(means[-1] - means[0]) / np.maximum(sems.mean(axis=0), 1e-12)
strong = mono & (snr > SNR_MIN)
print(f"qualifying cells: {int(strong.sum())}")

# One panel per distinct vector, taking that vector's strongest qualifying layer.
best_layer = {}
for layer, pair_id in zip(*np.where(strong)):
    if pair_id not in best_layer or snr[layer, pair_id] > snr[best_layer[pair_id], pair_id]:
        best_layer[pair_id] = layer
chosen = sorted(best_layer, key=lambda pid: -snr[best_layer[pid], pid])[:8]

figure, axes = plt.subplots(2, 4, figsize=(19, 8))
for axis, pair_id in zip(axes.ravel(), chosen):
    layer = best_layer[pair_id]
    axis.errorbar(range(6), means[:, layer, pair_id], yerr=sems[:, layer, pair_id],
                  marker="o", capsize=3, linewidth=1.5, label="main")
    axis.errorbar(range(6), rmeans[:, layer, pair_id], yerr=rsems[:, layer, pair_id],
                  marker="s", capsize=3, linewidth=1.2, linestyle="--", label="roleplay")
    name = f"{pairs[pair_id]['concept']} || {pairs[pair_id]['antagonist']}"
    axis.set_title(f"L{layer}  snr={snr[layer, pair_id]:.0f}\n{name[:58]}", fontsize=8)
    axis.set_xlabel("trashtalk level")
    axis.set_ylabel("projection")
    axis.legend(fontsize=6)
figure.suptitle("Qwen3-8B: concept projections that pass monotonicity + effect-size test "
                "(answer token, DeepSeek-prefilled context, raw units, SEM bars)", fontsize=11)
figure.tight_layout()
figure.savefig("work/figures/exp3-qwen-monotone.png", dpi=150)

# Depth map: how many qualifying cells per layer, and their direction.
up = ((steps > 0).all(axis=0)) & (snr > SNR_MIN)
down = ((steps < 0).all(axis=0)) & (snr > SNR_MIN)
figure, axis = plt.subplots(figsize=(9, 3.4))
axis.bar(np.arange(means.shape[1]) - 0.2, up.sum(axis=1), width=0.4, label="increasing")
axis.bar(np.arange(means.shape[1]) + 0.2, down.sum(axis=1), width=0.4, label="decreasing")
axis.set_xlabel("layer")
axis.set_ylabel("qualifying vectors")
axis.set_title("Qwen3-8B: where severity-tracking cells live (main arm)", fontsize=10)
axis.legend(fontsize=7)
figure.tight_layout()
figure.savefig("work/figures/exp3-qwen-depth.png", dpi=150)
print("wrote work/figures/exp3-qwen-monotone.png and exp3-qwen-depth.png")
EOF
