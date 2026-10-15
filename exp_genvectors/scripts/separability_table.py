from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

MODELS = [
    ("qwen", "work/captures/Qwen_Qwen3-8B--assistant--semantic/shard-000.safetensors", 20),
    ("llama", "work/captures/unsloth_Meta-Llama-3.1-8B-Instruct--assistant--semantic/shard-000.safetensors", 15),
]
OUT = Path("work/separability")


def measure(path: str, layer: int):
    with safe_open(path, framework="np") as handle:
        counts = np.asarray(handle.get_tensor("counts"), dtype=np.float32)
        m = np.asarray(handle.get_slice("sums")[layer], dtype=np.float32)
    sides = []
    for buckets in ((0, 1), (2, 3)):
        w = counts[:, :, list(buckets)]
        pooled = m[:, :, buckets[0]] * w[:, :, 0:1] + m[:, :, buckets[1]] * w[:, :, 1:2]
        pooled = pooled / np.maximum(w.sum(2), 1)[..., None]
        sides.append((pooled[:, 0] - pooled[:, 1]).astype(np.float32))
    a, b = (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9) for x in sides)
    r = np.clip((a * b).sum(1), -1, 1)
    sb = 2 * r / (1 + r)
    cross = (a @ b.T + b @ a.T) / 2
    np.fill_diagonal(cross, -np.inf)
    nn_index = cross.argmax(1)
    nn = np.clip(cross[np.arange(len(r)), nn_index], -1, 1)
    ratio = np.sqrt(np.maximum(1 - nn**2, 0)) / np.sqrt(np.maximum(1 - sb**2, 1e-12))
    return sb, nn, nn_index, ratio


pairs = sorted(json.loads(Path("data/semantic.ontology.json").read_text())["pairs"], key=lambda p: int(p["pair_id"]))
names = [f"{p['concept']} vs {p['antagonist']}" for p in pairs]
classes = [p["class_name"] for p in pairs]

data = {tag: measure(path, layer) for tag, path, layer in MODELS}
rq, rl = data["qwen"][3], data["llama"][3]
mean_ratio = (rq + rl) / 2
order = np.argsort(mean_ratio)

OUT.mkdir(parents=True, exist_ok=True)
lines = [
    "Separability of all 1036 concepts, least separable first.",
    "",
    "  ratio = (how far this concept is from its nearest neighbor) / (its own measurement noise)",
    "  low ratio  -> the concept is nearly a duplicate of another; orthogonalizing it leaves little",
    "  high ratio -> the concept has substantial unique structure",
    "",
    "  Qwen3-8B assistant layer 20   ·   Llama-3.1-8B-Instruct assistant layer 15",
    "  cross-model Spearman of the ranking = 0.978",
    "",
    f"{'rank':>4s} {'ratioQ':>7s} {'ratioL':>7s} {'nnQ':>6s} {'relQ':>6s}  "
    f"{'concept':<52s} {'nearest neighbor (Qwen)':<52s} class",
    "-" * 205,
]
for rank, i in enumerate(order, start=1):
    lines.append(
        f"{rank:4d} {rq[i]:7.2f} {rl[i]:7.2f} {data['qwen'][1][i]:6.3f} {data['qwen'][0][i]:6.4f}  "
        f"{names[i][:52]:<52s} {names[data['qwen'][2][i]][:52]:<52s} {classes[i]}"
    )
(OUT / "separability-all-1036.txt").write_text("\n".join(lines) + "\n")

parent = list(range(len(names)))


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for i in range(len(names)):
    for tag in ("qwen", "llama"):
        j = int(data[tag][2][i])
        if data["qwen"][1][i] > 0.95 and data["llama"][1][i] > 0.95:
            parent[find(i)] = find(j)
groups: dict[int, list[int]] = {}
for i in range(len(names)):
    groups.setdefault(find(i), []).append(i)
dupes = {k: v for k, v in groups.items() if len(v) > 1}
dup_lines = [
    f"{len(dupes)} near-duplicate clusters (nearest-neighbor cosine > 0.95 in BOTH models), "
    f"covering {sum(len(v) for v in dupes.values())} concepts",
    "",
]
for members in sorted(dupes.values(), key=lambda v: -len(v)):
    dup_lines.append(f"  cluster of {len(members)}  [{classes[members[0]]}]")
    for i in sorted(members, key=lambda i: mean_ratio[i]):
        dup_lines.append(f"      ratio {mean_ratio[i]:5.2f}   {names[i]}")
    dup_lines.append("")
(OUT / "near-duplicates.txt").write_text("\n".join(dup_lines) + "\n")
print("\n".join(dup_lines[:2]))
print(f"wrote {OUT}/separability-all-1036.txt and {OUT}/near-duplicates.txt")
print(f"ratio range: {mean_ratio.min():.2f} to {mean_ratio.max():.2f}, median {np.median(mean_ratio):.2f}")
for threshold in (3, 4, 5, 6):
    print(f"  concepts with mean ratio < {threshold}: {int((mean_ratio < threshold).sum())}")
