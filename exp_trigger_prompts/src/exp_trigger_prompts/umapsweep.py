from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq
from safetensors import safe_open

from .probe import VEC_ROOT, VEC_TAG

MODEL = "Qwen/Qwen3-8B"
LAYERS = list(range(12, 25, 2))
VARIANTS = ("raw", "centered", "nopc1")
NNEIGH = (15, 30)
MIN_DIST = 0.1
K_MARK = 10
UNIVERSAL_DF = 0.25
K_PURITY = 5
K_RETAIN = 10
SEED = 0


def promoted(V: np.ndarray, W: np.ndarray, gamma: np.ndarray, tok: Any) -> list[list[str]]:
    """Top-K logit-lens promoted words per concept, chunked so the [n, vocab] score block stays small."""
    words: list[list[str]] = []
    for s in range(0, V.shape[0], 64):
        scores = (V[s : s + 64] * gamma) @ W.T
        top = np.argpartition(scores, -K_MARK * 2, axis=1)[:, -K_MARK * 2 :]
        for bi in range(scores.shape[0]):
            seen: list[str] = []
            for tid in top[bi][np.argsort(-scores[bi, top[bi]])]:
                w = str(tok.decode([int(tid)])).strip().lower()
                if w and "�" not in w and w not in seen:
                    seen.append(w)
                if len(seen) == K_MARK:
                    break
            words.append(seen)
    return words


def vocab_sharing(words: list[list[str]]) -> dict[str, Any]:
    """How much promoted vocabulary do concepts share? mean pairwise Jaccard + universal-token load."""
    sets = [set(w) for w in words]
    n = len(sets)
    df: dict[str, int] = {}
    for s in sets:
        for w in s:
            df[w] = df.get(w, 0) + 1
    universal = {w for w, c in df.items() if c >= UNIVERSAL_DF * n}
    rng = np.random.default_rng(SEED)
    a, b = rng.integers(0, n, 20000), rng.integers(0, n, 20000)
    jac = [len(sets[i] & sets[j]) / len(sets[i] | sets[j]) for i, j in zip(a, b, strict=True) if i != j]
    load = [len(s & universal) / max(len(s), 1) for s in sets]
    return {
        "mean_jaccard": round(float(np.mean(jac)), 4),
        "universal_frac": round(float(np.mean(load)), 4),
        "n_universal_tokens": len(universal),
        "top_universal": [w for w, _ in sorted(df.items(), key=lambda kv: -kv[1])[:20]],
    }


def knn(X: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest neighbors; inputs are unit-normalized, so cosine and euclidean agree."""
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    return idx


def main() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(var, os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    import umap
    from sklearn.manifold import trustworthiness

    meta = pq.read_table(VEC_ROOT / VEC_TAG[MODEL] / "pairs.parquet").to_pydict()
    classes = np.array(meta["class_name"])
    with safe_open(VEC_ROOT / VEC_TAG[MODEL] / "diff.safetensors", framework="pt") as h:
        diff = h.get_tensor("diff").float().numpy()
    print(f"diff {diff.shape}, {len(set(classes))} ontology classes", flush=True)

    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoTokenizer

    cached = try_to_load_from_cache(MODEL, "model.safetensors.index.json")
    if not isinstance(cached, str):
        raise SystemExit(f"{MODEL} index not in the HF cache; fetch it on the login node first")
    root = Path(cached).parent
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    head = "lm_head.weight" if "lm_head.weight" in index else "model.embed_tokens.weight"
    shards = {}
    for name in (head, "model.norm.weight"):
        with safe_open(root / index[name], framework="pt") as h:
            shards[name] = h.get_tensor(name).float().numpy()
    W, gamma, tok = shards[head], shards["model.norm.weight"], AutoTokenizer.from_pretrained(MODEL)
    ref = LAYERS[len(LAYERS) // 2]
    Vr = diff[ref] / np.linalg.norm(diff[ref], axis=1, keepdims=True)
    Vc = Vr - Vr.mean(0)
    Vc /= np.linalg.norm(Vc, axis=1, keepdims=True)
    marker: dict[str, Any] = {
        "ref_layer": ref,
        "raw": vocab_sharing(promoted(Vr, W, gamma, tok)),
        "centered": vocab_sharing(promoted(Vc, W, gamma, tok)),
    }
    del W
    print(f"\nL{ref} promoted-vocabulary sharing (top-{K_MARK}):", flush=True)
    for k in ("raw", "centered"):
        m = marker[k]
        print(
            f"  {k:9s} mean pairwise Jaccard={m['mean_jaccard']:.4f}  universal-token frac="
            f"{m['universal_frac']:.4f}  ({m['n_universal_tokens']} universal tokens)",
            flush=True,
        )
        print(f"            most shared: {', '.join(m['top_universal'][:12])}", flush=True)

    out: dict[str, Any] = {
        "model": MODEL,
        "layers": LAYERS,
        "variants": list(VARIANTS),
        "n_neighbors": list(NNEIGH),
        "min_dist": MIN_DIST,
        "seed": SEED,
        "concept": list(meta["concept"]),
        "class_name": list(classes),
        "marker_check": marker,
        "runs": {},
    }
    print("\nlayer var       meancos  purity_full  |  nn  purity_2d  retain@10  trust", flush=True)
    for L in LAYERS:
        U = diff[L] / np.linalg.norm(diff[L], axis=1, keepdims=True)
        C = U - U.mean(0)
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        pc1 = np.linalg.svd(C, full_matrices=False)[2][0]
        N = C - np.outer(C @ pc1, pc1)
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        for var, X in zip(VARIANTS, (U, C, N), strict=True):
            S = X @ X.T
            meancos = float((S.sum() - len(X)) / (len(X) * (len(X) - 1)))
            full_p = knn(X, K_PURITY)
            full_r = knn(X, K_RETAIN)
            pur_full = float(np.mean(classes[full_p] == classes[:, None]))
            for nn in NNEIGH:
                emb = umap.UMAP(n_neighbors=nn, min_dist=MIN_DIST, metric="cosine", random_state=SEED).fit_transform(X)
                E = cast("np.ndarray", emb).astype(np.float64)
                Ez = (E - E.mean(0)) / (E.std(0) + 1e-9)
                d2 = ((Ez[:, None, :] - Ez[None, :, :]) ** 2).sum(-1)
                np.fill_diagonal(d2, np.inf)
                e_p = np.argpartition(d2, K_PURITY, axis=1)[:, :K_PURITY]
                e_r = np.argpartition(d2, K_RETAIN, axis=1)[:, :K_RETAIN]
                retain = float(np.mean([len(set(a) & set(b)) / K_RETAIN for a, b in zip(full_r, e_r, strict=True)]))
                tw = float(trustworthiness(X, E, n_neighbors=K_RETAIN, metric="cosine"))
                key = f"L{L}-{var}-nn{nn}"
                out["runs"][key] = {
                    "layer": L,
                    "variant": var,
                    "n_neighbors": nn,
                    "mean_cos": round(meancos, 4),
                    "purity_full": round(pur_full, 4),
                    "purity_2d": round(float(np.mean(classes[e_p] == classes[:, None])), 4),
                    "retain_at_10": round(retain, 4),
                    "trustworthiness": round(tw, 4),
                    "xy": [[round(float(x), 3), round(float(y), 3)] for x, y in E],
                }
                r = out["runs"][key]
                print(
                    f"  L{L:2d} {var:9s} {meancos:+.3f}    {pur_full:.3f}     |  {nn:2d}   "
                    f"{r['purity_2d']:.3f}      {retain:.3f}     {tw:.3f}",
                    flush=True,
                )

    Path("work/umapsweep").mkdir(parents=True, exist_ok=True)
    Path("work/umapsweep/umapsweep.json").write_text(json.dumps(out, ensure_ascii=False))
    print("\nwrote work/umapsweep/umapsweep.json  (plot locally)")
    print("blob check: raw mean_cos >> centered mean_cos means the common mode was the blob;")
    print("structure check: purity_2d/retain rising from raw -> centered means it was hiding the classes.")


if __name__ == "__main__":
    main()
