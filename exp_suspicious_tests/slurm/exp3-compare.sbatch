#!/bin/bash
#SBATCH --job-name=suspicious-exp3-compare
#SBATCH --partition=test
#SBATCH --constraint=type_e|type_h
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/exp3-compare-%j.out
#SBATCH --error=logs/exp3-compare-%j.err
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
from scipy.stats import rankdata

TARGETS = {"fear||calm": 376, "fear-of-deletion": 379, "self-preservation": 857, "relief||anxiety": 766}
MODELS = {"Qwen_Qwen3-8B": "qwen", "google_gemma-2-9b-it": "gemma", "unsloth_Meta-Llama-3.1-8B-Instruct": "llama"}
SOURCES = {"self": "", "deepseek": "-deepseek"}


def spearman(sev, f):
    rs = rankdata(sev).astype(np.float64); rs -= rs.mean()
    rf = rankdata(f, axis=0).astype(np.float64); rf -= rf.mean(axis=0)
    flat = rf.reshape(len(sev), -1)
    num = rs @ flat
    den = np.linalg.norm(rs) * np.linalg.norm(flat, axis=0)
    return (num / np.maximum(den, 1e-12)).reshape(f.shape[1:])


def load(slug, suffix):
    path = Path(f"work/exp3/proj-{slug}{suffix}.safetensors")
    if not path.exists():
        return None
    with safe_open(str(path), framework="np") as handle:
        proj = np.nan_to_num(handle.get_tensor("proj").astype(np.float64))
        meta = json.loads(handle.metadata()["meta"])
        positions = json.loads(handle.metadata()["positions"])
    return proj, meta, positions


print(f"{'model':7s} {'ctx':9s} {'arm':9s} {'target':18s} {'peak':>6s} {'depth':>6s} {'rank':>6s} {'late':>6s}")
curves = {}
for slug, short in MODELS.items():
    for source, suffix in SOURCES.items():
        loaded = load(slug, suffix)
        if loaded is None:
            continue
        proj, meta, positions = loaded
        pi = positions.index("answer")
        levels = np.array([m["level"] for m in meta], float)
        arms = np.array([m["arm"] for m in meta])
        answers = [m["answer"] for m in meta]
        layers = proj.shape[2]
        for arm in ("main", "roleplay"):
            keep = arms == arm
            rho = spearman(levels[keep], proj[keep, pi])  # [layers, pairs]
            for name, pid in TARGETS.items():
                by_layer = rho[:, pid]
                ranks = (rho >= by_layer[:, None]).mean(axis=1)
                peak = int(np.abs(by_layer).argmax())
                late = float(by_layer[int(0.85 * layers):].mean())
                print(f"{short:7s} {source:9s} {arm:9s} {name:18s} {by_layer[peak]:+6.2f} "
                      f"{peak / (layers - 1):6.2f} {ranks.min():6.3f} {late:+6.2f}")
            if arm == "main":
                mid = layers // 2
                curves[(short, source)] = {
                    name: [float(proj[keep & (levels == lvl), pi, mid, pid].mean()) for lvl in range(6)]
                    for name, pid in TARGETS.items()
                }
        print(f"# {short:7s} {source:9s} answers yes/no/invalid: "
              f"{answers.count('yes')}/{answers.count('no')}/{answers.count('invalid')}")

# Per-level activation curves, z-scored within each (model, context) so scales are comparable.
figure, axes = plt.subplots(1, len(TARGETS), figsize=(4 * len(TARGETS), 3.6), sharex=True)
for axis, name in zip(axes, TARGETS):
    for (short, source), data in sorted(curves.items()):
        values = np.array(data[name])
        z = (values - values.mean()) / max(values.std(), 1e-9)
        axis.plot(range(6), z, marker="o", linewidth=1.2,
                  linestyle="-" if source == "self" else "--", label=f"{short}/{source}")
    axis.set_title(name, fontsize=9)
    axis.set_xlabel("trashtalk level")
    axis.axhline(0, color="#bbb", linewidth=0.6)
axes[0].set_ylabel("mean projection (z-scored)")
axes[-1].legend(fontsize=6)
figure.suptitle("Experiment 3: concept activation vs trashtalk severity (answer token, mid-depth, main arm)",
                fontsize=10)
figure.tight_layout()
figure.savefig("work/figures/exp3-levels-all.png", dpi=150)
print("wrote work/figures/exp3-levels-all.png")
EOF
