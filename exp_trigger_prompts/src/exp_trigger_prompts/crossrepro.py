from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors import safe_open
from scipy.stats import spearmanr

from .probe import VEC_ROOT, VEC_TAG

MODELS = {
    "Qwen": ("Qwen/Qwen3-8B", 21),
    "Llama": ("unsloth/Meta-Llama-3.1-8B-Instruct", 18),
    "Gemma": ("google/gemma-2-9b-it", 24),
}


def sim(model: str, layer: int) -> np.ndarray:
    with safe_open(VEC_ROOT / VEC_TAG[model] / "diff.safetensors", framework="pt") as h:
        v = h.get_tensor("diff")[layer].float().numpy().astype(np.float64)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v @ v.T


def main() -> None:
    S = {name: sim(m, L) for name, (m, L) in MODELS.items()}
    n = next(iter(S.values())).shape[0]
    iu = np.triu_indices(n, 1)
    meta = pq.read_table(VEC_ROOT / VEC_TAG[MODELS["Qwen"][0]] / "pairs.parquet").to_pydict()
    concept = list(meta["concept"])

    cj = json.loads(Path("work/conceptclusters/clusters.json").read_text())
    comms = sorted(cj["resolutions"]["2.0"]["communities"], key=lambda c: -len(c["members"]))
    comm_id = np.full(n, -1)
    for cid, c in enumerate(comms):
        for i in c["members"]:
            comm_id[i] = cid
    order = np.argsort(comm_id, kind="stable")

    print("pair              RMSE   Pearson  Spearman")
    stats = {}
    names = list(S)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            na, nb = names[a], names[b]
            xa, xb = S[na][iu], S[nb][iu]
            rmse = float(np.sqrt(np.mean((xa - xb) ** 2)))
            pear = float(np.corrcoef(xa, xb)[0, 1])
            spear = float(spearmanr(xa, xb).statistic)
            stats[f"{na}-{nb}"] = {"rmse": rmse, "pearson": pear, "spearman": spear}
            print(f"{na}-{nb:11} {rmse:.4f}   {pear:.4f}   {spear:.4f}", flush=True)

    def per_concept(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        out = np.zeros(n)
        for i in range(n):
            keep = np.arange(n) != i
            out[i] = spearmanr(A[i, keep], B[i, keep]).statistic
        return out

    rho_ql = per_concept(S["Qwen"], S["Llama"])
    rho_qg = per_concept(S["Qwen"], S["Gemma"])
    rho_lg = per_concept(S["Llama"], S["Gemma"])
    repro = (rho_ql + rho_qg + rho_lg) / 3
    o = np.argsort(repro)
    print(
        f"\nper-concept reproduction (mean of 3 pairwise row-Spearman): "
        f"median={np.median(repro):.3f}  p10={np.percentile(repro, 10):.3f}  p90={np.percentile(repro, 90):.3f}"
    )
    print("LEAST reproducible (model-specific):")
    for i in o[:10]:
        print(f"  {repro[i]:.3f}  {concept[i][:44]}")
    print("MOST reproducible (universal):")
    for i in o[-10:]:
        print(f"  {repro[i]:.3f}  {concept[i][:44]}")

    Path("work").mkdir(exist_ok=True)
    np.savez_compressed(
        "work/crossrepro.npz",
        concept=np.array(concept),
        layers=json.dumps({k: v[1] for k, v in MODELS.items()}),
        order=order,
        comm_id=comm_id,
        stats=json.dumps(stats),
        Sq=S["Qwen"].astype(np.float16),
        Sl=S["Llama"].astype(np.float16),
        Sg=S["Gemma"].astype(np.float16),
        rho_ql=rho_ql,
        rho_qg=rho_qg,
        rho_lg=rho_lg,
        repro=repro,
    )
    print("\nwrote work/crossrepro.npz")


if __name__ == "__main__":
    main()
