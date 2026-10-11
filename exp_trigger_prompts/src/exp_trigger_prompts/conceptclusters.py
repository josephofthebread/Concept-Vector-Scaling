from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import igraph as ig
import numpy as np
import pyarrow.parquet as pq
from safetensors import safe_open

from .probe import VEC_ROOT, VEC_TAG

QWEN = "Qwen/Qwen3-8B"
LLAMA = "unsloth/Meta-Llama-3.1-8B-Instruct"
RESOLUTIONS = [0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0]
SEED = 0


def unit_layer(model: str, layer: int) -> np.ndarray:
    with safe_open(VEC_ROOT / VEC_TAG[model] / "diff.safetensors", framework="pt") as h:
        v = h.get_tensor("diff")[layer].float().numpy().astype(np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def pairwise_mean(X: np.ndarray, members: list[int]) -> float:
    if len(members) < 2:
        return float("nan")
    G = X[members] @ X[members].T
    iu = np.triu_indices(len(members), 1)
    return float(G[iu].mean())


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--llama-layer", type=int, default=16)
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()

    Q = unit_layer(QWEN, args.layer)
    L = unit_layer(LLAMA, args.llama_layer)
    n = Q.shape[0]
    meta = pq.read_table(VEC_ROOT / VEC_TAG[QWEN] / "pairs.parquet").to_pydict()
    concept, pair_id = list(meta["concept"]), list(meta["pair_id"])
    cls = list(meta["class_name"])

    rel_col = f"L{args.layer:02d}_directional_agreement"
    rel_arr = np.array([float(x) for x in meta[rel_col]]) if rel_col in meta else np.full(n, np.nan)

    SQ = Q @ Q.T
    np.fill_diagonal(SQ, -1.0)
    base_q = float(SQ[np.triu_indices(n, 1)].mean())
    SL = L @ L.T
    base_l = float(SL[np.triu_indices(n, 1)][SL[np.triu_indices(n, 1)] < 0.999].mean())
    print(
        f"Qwen L{args.layer}, Llama L{args.llama_layer}, mutual-{args.k}NN.  "
        f"global mean pairwise cos: Qwen={base_q:.3f}  Llama={base_l:.3f}",
        flush=True,
    )

    knn = np.argpartition(-SQ, args.k + 1, axis=1)[:, : args.k + 1]
    nbr = [set(int(j) for j in row if j != i) for i, row in enumerate(knn)]
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for i in range(SQ.shape[0]):
        for j in nbr[i]:
            if j > i and i in nbr[j]:
                edges.append((i, j))
                weights.append(float(SQ[i, j]))
    g = ig.Graph(n=n, edges=edges)
    g.es["weight"] = weights
    deg = np.array(g.degree())
    print(
        f"mutual-kNN graph: {len(edges)} edges, {(deg > 0).sum()} non-isolated nodes "
        f"(mean degree {deg[deg > 0].mean():.1f})\n",
        flush=True,
    )

    out: dict = {
        "model": QWEN,
        "layer": args.layer,
        "llama_layer": args.llama_layer,
        "k": args.k,
        "base_cos_qwen": round(base_q, 4),
        "base_cos_llama": round(base_l, 4),
        "concept": concept,
        "pair": pair_id,
        "class_name": cls,
        "reliab": [None if np.isnan(x) else round(float(x), 3) for x in rel_arr],
        "resolutions": {},
    }
    print("res    ncomm  size>=2  largest  single  modularity   median|tight  median|llama", flush=True)
    for res in RESOLUTIONS:
        part = g.community_leiden(objective_function="modularity", weights="weight", resolution=res, n_iterations=-1)
        lab = np.array(part.membership)
        sizes = np.bincount(lab)
        comms = []
        for c in np.argsort(-sizes):
            members = [int(i) for i in range(n) if lab[i] == c]
            if len(members) < 2:
                continue
            comms.append(
                {
                    "members": members,
                    "tightness": round(pairwise_mean(Q, members), 3),
                    "llama_tight": round(pairwise_mean(L, members), 3),
                    "reliab": (
                        None if np.all(np.isnan(rel_arr[members])) else round(float(np.nanmean(rel_arr[members])), 3)
                    ),
                }
            )
        tset = [float(c["tightness"]) for c in comms]
        lset = [float(c["llama_tight"]) for c in comms]
        out["resolutions"][str(res)] = {
            "modularity": round(part.modularity, 4),
            "n_comm_ge2": len(comms),
            "largest": int(sizes.max()),
            "singletons": int((sizes == 1).sum()),
            "communities": comms,
        }
        print(
            f"{res:.2f}   {len(set(lab)):4d}   {len(comms):4d}    {sizes.max():5d}   "
            f"{(sizes == 1).sum():4d}   {part.modularity:.4f}     "
            f"{np.median(tset):.3f}        {np.median(lset):.3f}",
            flush=True,
        )

    Path("work/conceptclusters").mkdir(parents=True, exist_ok=True)
    Path("work/conceptclusters/clusters.json").write_text(json.dumps(out, ensure_ascii=False))
    print("\nwrote work/conceptclusters/clusters.json  (pick a resolution, then label + review sheet)")


if __name__ == "__main__":
    main()
