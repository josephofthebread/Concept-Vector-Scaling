from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

PATH = "work/captures/Qwen_Qwen3-8B--assistant--semantic/shard-000.safetensors"
ONTOLOGY = Path("data/semantic.ontology.json")


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def main():
    pairs = json.loads(ONTOLOGY.read_text())["pairs"]
    pairs.sort(key=lambda p: int(p["pair_id"]))
    n = len(pairs)
    with safe_open(PATH, framework="np") as handle:
        counts = np.asarray(handle.get_tensor("counts"), dtype=np.float32)
        nlay = handle.get_slice("sums").get_shape()[0]
        ratio = np.zeros((nlay, n), np.float32)
        sbv = np.zeros((nlay, n), np.float32)
        for layer in range(nlay):
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
            nn = np.clip(cross.max(1), -1, 1)
            noise = np.sqrt(np.maximum(1 - sb**2, 1e-12))
            residual = np.sqrt(np.maximum(1 - nn**2, 0))
            ratio[layer] = residual / noise
            sbv[layer] = sb
    np.savez("work/separability_percpt.npz", ratio=ratio, sb=sbv, layers=np.arange(nlay))
    print(f"wrote work/separability_percpt.npz ratio{ratio.shape}")


if __name__ == "__main__":
    main()
