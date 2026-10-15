import os
import sys

if "SLURM_JOB_ID" not in os.environ:
    sys.exit("Do not run this on the login node. Use sbatch")

import json
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

torch.set_num_threads(8)
G = Path("../exp_genvectors")
A = Path(".")
V = load_file(str(G / "work/vectors/unsloth_Meta-Llama-3.1-8B-Instruct--assistant--semantic/diff.safetensors"))[
    "diff"
].float()
R = V.shape[0]
Qs = []
for L in range(R):
    X = V[L]
    m = X.mean(0)
    mh = m / m.norm()
    _, _, Wt = torch.linalg.svd(X - m, full_matrices=False)
    Q, _ = torch.linalg.qr(torch.cat([mh[None], Wt[:16]], 0).T)
    Qs.append(Q)


def xcos(vl, L):
    s = (vl @ Qs[L]) @ Qs[L].T
    r = vl - s
    off = ~np.eye(vl.shape[0], dtype=bool)
    f = lambda X: torch.nn.functional.cosine_similarity(X[:, None], X[None], dim=-1).numpy()[off].mean()
    return f(vl), f(s), f(r), (r.norm(dim=-1) ** 2 / vl.norm(dim=-1) ** 2).mean().item()


pres = ["assistant", "user", "named_assistant", "paired_assistant", "named_assistant_decl"]
VP = torch.stack(
    [
        load_file(str(G / f"work/vectors/unsloth_Meta-Llama-3.1-8B-Instruct--{p}--semantic/diff.safetensors"))[
            "diff"
        ].float()
        for p in pres
    ]
)
resP = np.zeros((V.shape[1], R, 4))
for L in range(R):
    for k in range(V.shape[1]):
        resP[k, L] = xcos(VP[:, L, k], L)
print("presentations (5):")
for i, nm in enumerate(("full", "stance", "resid", "rshare")):
    print(f"  {nm:7s}", " ".join(f"{np.median(resP[:, L, i]):.2f}" for L in range(R)))
files = sorted((G / "work/perstory-llama").glob("pair-*.safetensors"))
resG = np.zeros((len(files), R, 4))
resE = np.zeros((len(files), R, 4))
resL = np.zeros((len(files), R, 4))
for i, f in enumerate(files):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        rows = json.loads(json.loads(fh.read(n))["__metadata__"]["manifest"])["rows"]
    genre = np.array([r["genre_id"] for r in rows])
    gen = np.array([r["generator_id"] for r in rows])
    lang = np.array([r["language_id"] for r in rows])
    D = load_file(str(f))["diff"]
    for L in range(R):
        X = D[:, L].float()
        resG[i, L] = xcos(torch.stack([X[genre == g].mean(0) for g in range(10)]), L)
        resE[i, L] = xcos(torch.stack([X[gen == g].mean(0) for g in range(2)]), L)
        resL[i, L] = xcos(torch.stack([X[lang == g].mean(0) for g in range(5)]), L)
    print(i, flush=True)
np.savez(str(A / "work-Meta-Llama-3.1-8B-Instruct/invprof.npz"), pres=resP, genre=resG, generator=resE, lang=resL)
for nm, arr in (
    ("genres (10, ~100 stories each)", resG),
    ("generators (2, 500 each)", resE),
    ("languages (5, 200 each)", resL),
):
    print(nm + ":")
    for i, q in enumerate(("full", "stance", "resid", "rshare")):
        print(f"  {q:7s}", " ".join(f"{np.median(arr[:, L, i]):.2f}" for L in range(R)))
