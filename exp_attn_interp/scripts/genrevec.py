import json
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

torch.set_num_threads(8)
G = Path("../exp_genvectors")
A = Path(".")
K, H, L = 1036, 4096, 22
out = {f"g{g}": torch.zeros(K, H) for g in range(10)}
pairs = []
for f in sorted((G / "work/perstory").glob("pair-*.safetensors")):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        meta = json.loads(fh.read(n))["__metadata__"]
    genre = np.array([r["genre_id"] for r in json.loads(meta["manifest"])["rows"]])
    k = int(f.stem.split("-")[1])
    pairs.append(k)
    X = load_file(str(f))["diff"][:, L].float()
    for g in range(10):
        out[f"g{g}"][k] = X[genre == g].mean(0)
    print(k, flush=True)
save_file(
    {**{k: v.contiguous() for k, v in out.items()}, "pairs": torch.tensor(pairs)},
    str(A / f"work/genrevec-L{L}.safetensors"),
)
print("done")
