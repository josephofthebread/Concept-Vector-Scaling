from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from safetensors import safe_open

FIELDS = ["language", "genre", "generator"]
try:
    from exp_gendata.config import GENRES, LANGUAGES

    NAMES = {"language": LANGUAGES, "genre": GENRES, "generator": ["gen0", "gen1"]}
except Exception:
    NAMES = {}


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else float("nan")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--tag", default="Qwen3-8B")
    args = parser.parse_args()

    results: dict[str, list[dict]] = {}
    with safe_open(str(args.capture), framework="np") as f:
        keys = set(f.keys())
        for field in FIELDS:
            if f"{field}_sum" not in keys:
                continue
            mu = f.get_slice(f"{field}_sum")[args.layer].astype(np.float64)
            cnt = f.get_tensor(f"{field}_count").astype(np.float64)
            P, _, G, _H = mu.shape
            rows: list[dict] = []
            for p in range(P):
                for g in range(G):
                    if cnt[p, 0, g] < 1 or cnt[p, 1, g] < 1:
                        continue
                    others = [h for h in range(G) if h != g and cnt[p, 0, h] >= 1 and cnt[p, 1, h] >= 1]
                    if not others:
                        continue
                    wp, wa = cnt[p, 0, others], cnt[p, 1, others]
                    v_out = np.average(mu[p, 0, others], axis=0, weights=wp) - np.average(
                        mu[p, 1, others], axis=0, weights=wa
                    )
                    v_in = mu[p, 0, g] - mu[p, 1, g]
                    v_all = np.average(mu[p, 0], axis=0, weights=cnt[p, 0]) - np.average(
                        mu[p, 1], axis=0, weights=cnt[p, 1]
                    )
                    rows.append(
                        {
                            "pair": p,
                            "cell": g,
                            "n": int(cnt[p, 0, g]),
                            "cos_out_in": cos(v_out, v_in),
                            "cos_all_in": cos(v_all, v_in),
                            "_vout": v_out,
                            "_vin": v_in,
                        }
                    )
            results[field] = rows
            oo = np.array([r["cos_out_in"] for r in rows if not np.isnan(r["cos_out_in"])])
            base = []
            for g in sorted({r["cell"] for r in rows}):
                gr = [r for r in rows if r["cell"] == g]
                vo = np.stack([r["_vout"] for r in gr])
                vi = np.stack([r["_vin"] for r in gr])
                perm = (np.arange(len(gr)) + 1) % len(gr)
                for a, b in zip(vo, vi[perm]):
                    base.append(cos(a, b))
            base = np.array([x for x in base if not np.isnan(x)])
            for r in rows:
                r.pop("_vout", None)
                r.pop("_vin", None)
            print(f"\n=== {field} leave-one-out direction transfer ({len(rows)} cells, L{args.layer}) ===")
            print(
                f"  cos(other-cells vector, held-out cell vector): median {np.median(oo):.3f} "
                f"mean {oo.mean():.3f}  (1=identical, ~0=specific)"
            )
            print(
                f"  cross-concept baseline (shuffled): median {np.median(base):.3f}  "
                f"-> concept-specific transfer margin {np.median(oo) - np.median(base):+.3f}"
            )
            nm = NAMES.get(field)
            for g in sorted({r["cell"] for r in rows}):
                gc = [r["cos_out_in"] for r in rows if r["cell"] == g and not np.isnan(r["cos_out_in"])]
                label = nm[g] if nm and g < len(nm) else str(g)
                print(f"    cell {g} ({label}): median cos {np.median(gc):.3f}  (n={len(gc)})")

    out = Path("work/oodcos")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"oodcos-{args.tag}.json").write_text(json.dumps({"layer": args.layer, "results": results}))
    print(f"\nwrote {out}/oodcos-{args.tag}.json")


if __name__ == "__main__":
    main()
