from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

CAPTURES = {
    "Qwen3-8B assistant": "work/captures/Qwen_Qwen3-8B--assistant--semantic/shard-000.safetensors",
    "Llama-3.1-8B-Instruct assistant": "work/captures/unsloth_Meta-Llama-3.1-8B-Instruct--assistant--semantic/shard-000.safetensors",
}
ONTOLOGY = Path("data/semantic.ontology.json")


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


pairs = json.loads(ONTOLOGY.read_text())["pairs"]
pairs.sort(key=lambda p: int(p["pair_id"]))
names = [f"{p['concept']} vs {p['antagonist']}" for p in pairs]
classes = np.array([p["class_name"] for p in pairs])

results = {}
for label, path in CAPTURES.items():
    with safe_open(path, framework="np") as handle:
        counts = np.asarray(handle.get_tensor("counts"), dtype=np.float32)
        layers = handle.get_slice("sums").get_shape()[0]
        print(f"\n{'=' * 78}\n{label}   ({layers} layers, {len(names)} concepts)\n{'=' * 78}", flush=True)

        print(f"{'layer':>5s} {'median r':>9s} {'median nn':>10s} {'median ratio':>13s} {'ratio<2':>8s}")
        best, per_layer = (-1.0, None), []
        for layer in range(layers):
            m = np.asarray(handle.get_slice("sums")[layer], dtype=np.float32)
            split = []
            for buckets in ((0, 1), (2, 3)):
                weight = counts[:, :, list(buckets)]
                pooled = m[:, :, buckets[0]] * weight[:, :, 0:1] + m[:, :, buckets[1]] * weight[:, :, 1:2]
                pooled = pooled / np.maximum(weight.sum(2), 1)[..., None]
                split.append((pooled[:, 0] - pooled[:, 1]).astype(np.float32))
            a, b = split[0], split[1]
            a, b = unit(a), unit(b)
            r = np.clip((a * b).sum(1), -1, 1)
            sb = 2 * r / (1 + r)
            cross = (a @ b.T + b @ a.T) / 2
            np.fill_diagonal(cross, -np.inf)
            nn_index = cross.argmax(1)
            nn = np.clip(cross[np.arange(len(names)), nn_index], -1, 1)
            noise = np.sqrt(np.maximum(1 - sb**2, 1e-12))
            residual = np.sqrt(np.maximum(1 - nn**2, 0))
            ratio = residual / noise
            per_layer.append((layer, sb, nn, nn_index, ratio))
            print(
                f"L{layer:<4d} {np.median(sb):9.4f} {np.median(nn):10.4f} "
                f"{np.median(ratio):13.2f} {int((ratio < 2).sum()):8d}",
                flush=True,
            )
            if np.median(ratio) > best[0]:
                best = (float(np.median(ratio)), layer)

        layer, sb, nn, nn_index, ratio = per_layer[best[1]]
        results[label] = {"layer": layer, "ratio": ratio, "sb": sb, "nn": nn, "nn_index": nn_index}
        print(f"\n--- layer {layer} (highest median ratio) ---")
        order = np.argsort(-ratio)
        print("\nMOST separable (orthogonalization leaves real signal):")
        print(f"{'ratio':>6s} {'rel':>7s} {'nn cos':>7s}  concept  ->  nearest neighbor")
        for i in order[:20]:
            print(f"{ratio[i]:6.1f} {sb[i]:7.4f} {nn[i]:7.4f}  {names[i][:44]:44s} -> {names[nn_index[i]][:40]}")
        print("\nLEAST separable (orthogonalization would distill noise):")
        for i in order[-20:]:
            print(f"{ratio[i]:6.1f} {sb[i]:7.4f} {nn[i]:7.4f}  {names[i][:44]:44s} -> {names[nn_index[i]][:40]}")

        by_class: dict[str, list[float]] = {}
        for cls, value in zip(classes, ratio, strict=True):
            by_class.setdefault(cls, []).append(float(value))
        ranked = sorted(by_class.items(), key=lambda kv: -np.median(kv[1]))
        print(f"\n{len(ranked)} classes, by median ratio")
        print("  most separable classes:")
        for cls, values in ranked[:10]:
            print(f"    {np.median(values):6.1f}  (n={len(values):3d})  {cls}")
        print("  least separable classes:")
        for cls, values in ranked[-10:]:
            print(f"    {np.median(values):6.1f}  (n={len(values):3d})  {cls}")

labels = list(results)
if len(labels) == 2:
    x, y = results[labels[0]]["ratio"], results[labels[1]]["ratio"]
    print(f"\n{'=' * 78}")
    ra, rb = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    print(
        f"cross-model agreement of the separability ranking: Spearman = {float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb))):.3f}"
    )
    print("(high = which concepts are separable is a property of the concepts, not of one model)")
