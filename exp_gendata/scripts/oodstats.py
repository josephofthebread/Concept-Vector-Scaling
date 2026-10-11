from __future__ import annotations

import json
import math
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from safetensors import safe_open

from exp_gendata.artifacts import read_capture

FIELDS = ["language", "genre", "generator"]


def auc_gauss(mu_p: np.ndarray, mu_a: np.ndarray, sig_p: np.ndarray, sig_a: np.ndarray, v: np.ndarray) -> float:
    """Gaussian AUROC of pole vs antagonist projected on direction v."""
    gap = float(v @ mu_p - v @ mu_a)
    var = float(v @ sig_p @ v + v @ sig_a @ v)
    if var <= 0:
        return float("nan")
    return float(0.5 * (1.0 + np.vectorize(math.erf)(np.array(gap / math.sqrt(var)) / math.sqrt(2.0))))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0, help="0 = mid depth (num_layers//2)")
    parser.add_argument("--tag", default="Qwen3-8B")
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    if args.inspect:
        with safe_open(str(args.capture), framework="np") as f:
            for k in f.keys():
                print(f"{k:24s} {f.get_tensor(k).shape} {f.get_tensor(k).dtype}")
        return

    tensors, _meta = read_capture(args.capture)
    sums = tensors["sums"].astype(np.float64)
    moments = tensors["moments"].astype(np.float64)
    counts = np.maximum(tensors["counts"].astype(np.float64), 1.0)
    nL = sums.shape[0]
    layer = args.layer or nL // 2
    s = sums[layer] if sums.ndim == 4 else sums[layer].sum(axis=-2)
    mo = moments[layer] if moments.ndim == 5 else moments[layer].sum(axis=-3)
    ct = counts[layer] if counts.ndim == 3 else counts[layer].sum(axis=-1)
    pooled_mu = s / ct[..., None]
    cov = mo / ct[..., None, None] - pooled_mu[..., :, None] * pooled_mu[..., None, :]

    results: dict[str, list[dict]] = {}
    with safe_open(str(args.capture), framework="np") as f:
        keys = set(f.keys())
        for field in FIELDS:
            if f"{field}_sum" not in keys:
                continue
            mu_cell = f.get_tensor(f"{field}_sum").astype(np.float64)[layer]
            cnt = f.get_tensor(f"{field}_count").astype(np.float64)
            cnt = cnt[layer] if cnt.ndim == 4 else cnt
            P, _, G, _H = mu_cell.shape
            rows: list[dict] = []
            for p in range(P):
                sig_p, sig_a = cov[p, 0], cov[p, 1]
                for g in range(G):
                    if cnt[p, 0, g] < 1 or cnt[p, 1, g] < 1:
                        continue
                    others = [h for h in range(G) if h != g and cnt[p, 0, h] >= 1 and cnt[p, 1, h] >= 1]
                    if not others:
                        continue
                    wp = cnt[p, 0, others]
                    wa = cnt[p, 1, others]
                    mp_rest = np.average(mu_cell[p, 0, others], axis=0, weights=wp)
                    ma_rest = np.average(mu_cell[p, 1, others], axis=0, weights=wa)
                    v_ood = mp_rest - ma_rest
                    v_id = mu_cell[p, 0, g] - mu_cell[p, 1, g]
                    rows.append(
                        {
                            "pair": p,
                            "cell": g,
                            "n_pole": int(cnt[p, 0, g]),
                            "ood": auc_gauss(mu_cell[p, 0, g], mu_cell[p, 1, g], sig_p, sig_a, v_ood),
                            "id": auc_gauss(mu_cell[p, 0, g], mu_cell[p, 1, g], sig_p, sig_a, v_id),
                        }
                    )
            results[field] = rows
            ood = np.array([r["ood"] for r in rows if not math.isnan(r["ood"])])
            idd = np.array([r["id"] for r in rows if not math.isnan(r["id"])])
            print(f"\n=== {field} leave-one-out ({len(rows)} cells, layer {layer}) ===")
            print(f"  OOD AUC: median {np.median(ood):.3f}  mean {ood.mean():.3f}")
            print(f"  ID  AUC: median {np.median(idd):.3f}  mean {idd.mean():.3f}")
            for g in sorted({r["cell"] for r in rows}):
                go = [r["ood"] for r in rows if r["cell"] == g and not math.isnan(r["ood"])]
                print(f"    cell {g}: OOD median {np.median(go):.3f}  (n={len(go)})")

    out = Path("work/oodstats")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"oodstats-{args.tag}.json").write_text(json.dumps({"layer": layer, "results": results}))
    print(f"\nwrote {out}/oodstats-{args.tag}.json")


if __name__ == "__main__":
    main()
