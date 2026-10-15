from __future__ import annotations

import numpy as np
from safetensors import safe_open

PATH = "work/captures/Qwen_Qwen3-8B--assistant--semantic/shard-000.safetensors"


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def cos_rows(a, b):
    return np.clip((unit(a) * unit(b)).sum(1), -1, 1)


def main():
    with safe_open(PATH, framework="np") as h:
        gcount = np.asarray(h.get_tensor("generator_count"), np.float32)
        bcount = np.asarray(h.get_tensor("counts"), np.float32)
        nlay = h.get_slice("generator_sum").get_shape()[0]
        n = gcount.shape[0]
        genagree = np.zeros((nlay, n), np.float32)
        reliab = np.zeros((nlay, n), np.float32)
        for L in range(nlay):
            gs = np.asarray(h.get_slice("generator_sum")[L], np.float32)
            vg = []
            for g in range(2):
                c = gs[:, 0, g] / np.maximum(gcount[:, 0, g : g + 1], 1)
                a = gs[:, 1, g] / np.maximum(gcount[:, 1, g : g + 1], 1)
                vg.append((c - a).astype(np.float32))
            genagree[L] = cos_rows(vg[0], vg[1])
            bs = np.asarray(h.get_slice("sums")[L], np.float32)
            halves = []
            for buckets in ((0, 1), (2, 3)):
                w = bcount[:, :, list(buckets)]
                pooled = bs[:, :, buckets[0]] * w[:, :, 0:1] + bs[:, :, buckets[1]] * w[:, :, 1:2]
                pooled = pooled / np.maximum(w.sum(2), 1)[..., None]
                halves.append((pooled[:, 0] - pooled[:, 1]).astype(np.float32))
            reliab[L] = cos_rows(halves[0], halves[1])
    np.savez("work/generator_bias.npz", genagree=genagree, reliab=reliab, layers=np.arange(nlay))
    print(f"wrote work/generator_bias.npz genagree{genagree.shape}")


if __name__ == "__main__":
    main()
