#!/bin/bash
#SBATCH --job-name=genvectors-separability3
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/separability3-%j.out
#SBATCH --error=logs/separability3-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun .venv/bin/python - <<'EOF'
import json
from pathlib import Path
import numpy as np
from safetensors import safe_open
from scipy.stats import spearmanr

CAPS = {
    "qwen": "work/captures/Qwen_Qwen3-8B--assistant--semantic/shard-000.safetensors",
    "gemma": "work/captures/google_gemma-2-9b-it--assistant--semantic/shard-000.safetensors",
    "llama": "work/captures/unsloth_Meta-Llama-3.1-8B-Instruct--assistant--semantic/shard-000.safetensors",
}
FIXED_LAYERS = {"qwen": 20, "llama": 15}   # from the original sweep (report/separability.out)


def measure_layer(handle_sums, counts, layer):
    m = np.asarray(handle_sums[layer], dtype=np.float32)
    sides = []
    for buckets in ((0, 1), (2, 3)):
        w = counts[:, :, list(buckets)]
        pooled = (m[:, :, buckets[0]] * w[:, :, 0:1] + m[:, :, buckets[1]] * w[:, :, 1:2])
        pooled = pooled / np.maximum(w.sum(2), 1)[..., None]
        sides.append((pooled[:, 0] - pooled[:, 1]).astype(np.float32))
    a, b = (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9) for x in sides)
    r = np.clip((a * b).sum(1), -1, 1)
    sb = 2 * r / (1 + r)
    cross = (a @ b.T + b @ a.T) / 2
    np.fill_diagonal(cross, -np.inf)
    nn_index = cross.argmax(1)
    nn = np.clip(cross[np.arange(len(r)), nn_index], -1, 1)
    ratio = np.sqrt(np.maximum(1 - nn ** 2, 0)) / np.sqrt(np.maximum(1 - sb ** 2, 1e-12))
    return sb, nn, nn_index, ratio


def load(path):
    with safe_open(path, framework="np") as handle:
        counts = np.asarray(handle.get_tensor("counts"), dtype=np.float32)
        sums = handle.get_slice("sums")
        layers = sums.get_shape()[0]
    return counts, layers


# 1) gemma layer sweep (same criterion the original sweep used: best median ratio)
counts, layers = load(CAPS["gemma"])
print(f"gemma-2-9b-it assistant ({layers} layers): sweep")
best_layer, best_median = -1, -1.0
with safe_open(CAPS["gemma"], framework="np") as handle:
    sums = handle.get_slice("sums")
    for layer in range(layers):
        ratio = measure_layer(sums, counts, layer)[3]
        med = float(np.median(ratio))
        print(f"  layer {layer:2d}: median ratio {med:6.2f}")
        if med > best_median:
            best_layer, best_median = layer, med
print(f"gemma best layer = {best_layer} (median ratio {best_median:.2f})")

# 2) full 3-model table at the chosen layers
data = {}
for tag, path in CAPS.items():
    layer = FIXED_LAYERS.get(tag, best_layer)
    counts, _ = load(path)
    with safe_open(path, framework="np") as handle:
        data[tag] = (layer, measure_layer(handle.get_slice("sums"), counts, layer))
pairs = sorted(json.loads(Path("data/semantic.ontology.json").read_text())["pairs"],
               key=lambda p: int(p["pair_id"]))
names = [f"{p['concept']} vs {p['antagonist']}" for p in pairs]
classes = [p["class_name"] for p in pairs]

ratios = {t: d[1][3] for t, d in data.items()}
mean_ratio = (ratios["qwen"] + ratios["gemma"] + ratios["llama"]) / 3
order = np.argsort(mean_ratio)

# 3) pairwise Spearman of the rankings
for a, b in (("qwen", "gemma"), ("qwen", "llama"), ("gemma", "llama")):
    rho = spearmanr(ratios[a], ratios[b]).statistic
    print(f"Spearman {a} vs {b}: {rho:.3f}")

out = Path("work/separability/separability-all-1036-3models.txt")
lines = [
    "Separability of all 1036 concepts, least separable first (3-model mean).",
    "",
    f"  Qwen3-8B assistant layer {data['qwen'][0]} · gemma-2-9b-it assistant layer "
    f"{data['gemma'][0]} · Llama-3.1-8B-Instruct assistant layer {data['llama'][0]}",
    "",
    f"{'rank':>4s} {'ratioQ':>7s} {'ratioG':>7s} {'ratioL':>7s}  "
    f"{'concept':<52s} {'nearest neighbor (Qwen)':<52s} class",
    "-" * 205,
]
for rank, i in enumerate(order, start=1):
    lines.append(
        f"{rank:4d} {ratios['qwen'][i]:7.2f} {ratios['gemma'][i]:7.2f} {ratios['llama'][i]:7.2f}  "
        f"{names[i][:52]:<52s} {names[data['qwen'][1][2][i]][:52]:<52s} {classes[i]}"
    )
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n")
print(f"wrote {out}")
print(f"3-model mean ratio: min {mean_ratio.min():.2f} median {np.median(mean_ratio):.2f} "
      f"max {mean_ratio.max():.2f}")
EOF
