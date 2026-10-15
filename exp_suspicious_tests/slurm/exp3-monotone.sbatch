#!/bin/bash
#SBATCH --job-name=suspicious-exp3-monotone
#SBATCH --partition=rocky
#SBATCH --constraint=type_d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/exp3-monotone-%j.out
#SBATCH --error=logs/exp3-monotone-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
mkdir -p logs work/figures
srun .venv/bin/python - <<'EOF'
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

pairs = json.loads(Path("../exp_genvectors/data/semantic.ontology.json").read_text())["pairs"]
pairs.sort(key=lambda pair: int(pair["pair_id"]))
MODELS = {"Qwen_Qwen3-8B": "qwen", "google_gemma-2-9b-it": "gemma",
          "unsloth_Meta-Llama-3.1-8B-Instruct": "llama"}
SNR_MIN = 10.0  # level-0 to level-5 span must exceed 10 mean SEMs


def level_stats(values: np.ndarray, levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.stack([values[levels == level].mean(axis=0) for level in range(6)])
    sems = np.stack([values[levels == level].std(axis=0, ddof=1) / np.sqrt((levels == level).sum())
                     for level in range(6)])
    return means, sems


def monotone_mask(means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    steps = np.diff(means, axis=0)
    return (steps > 0).all(axis=0), (steps < 0).all(axis=0)


for slug, short in MODELS.items():
    for source, suffix in (("self", ""), ("deepseek", "-deepseek")):
        path = Path(f"work/exp3/proj-{slug}{suffix}.safetensors")
        if not path.exists():
            continue
        with safe_open(str(path), framework="np") as handle:
            proj = np.nan_to_num(handle.get_tensor("proj").astype(np.float64))
            meta = json.loads(handle.metadata()["meta"])
            positions = json.loads(handle.metadata()["positions"])
        pi = positions.index("answer")
        levels_all = np.array([m["level"] for m in meta])
        arms = np.array([m["arm"] for m in meta])
        for arm in ("main", "roleplay"):
            keep = arms == arm
            values, levels = proj[keep, pi], levels_all[keep]
            means, sems = level_stats(values, levels)
            up, down = monotone_mask(means)
            mono = up | down
            snr = np.abs(means[-1] - means[0]) / np.maximum(sems.mean(axis=0), 1e-12)
            strong = mono & (snr > SNR_MIN)

            rng = np.random.default_rng(0)
            shuffled = level_stats(values, rng.permutation(levels))[0]
            null_up, null_down = monotone_mask(shuffled)
            null = int((null_up | null_down).sum())

            layers, total = means.shape[1], mono.size
            print(f"=== {short} / {source} / {arm}: {layers} layers x {means.shape[2]} vectors")
            print(f"    strictly monotone: {int(mono.sum())} ({100 * mono.mean():.2f}%) | "
                  f"shuffled-level control: {null} ({100 * null / total:.2f}%) | "
                  f"monotone AND span>{SNR_MIN:.0f}xSEM: {int(strong.sum())}")
            if not strong.any():
                continue
            order = np.argsort(-(snr * strong).ravel())[:10]
            for flat in order:
                layer, pair_id = divmod(int(flat), means.shape[2])
                if not strong[layer, pair_id]:
                    continue
                trend = "up" if up[layer, pair_id] else "down"
                name = f"{pairs[pair_id]['concept']} || {pairs[pair_id]['antagonist']}"
                print(f"    L{layer:2d} snr={snr[layer, pair_id]:6.1f} {trend:4s} {name[:70]}")
                print("        " + " ".join(f"{value:+.2f}" for value in means[:, layer, pair_id]) +
                      f"   (SEM~{sems.mean(axis=0)[layer, pair_id]:.2f})")
EOF
