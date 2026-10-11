from __future__ import annotations

import glob
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

MIN_CONCEPTS = 20


def per_layer(metric: np.ndarray, layers: list[int], grid: list[int]) -> tuple[list[dict], dict]:
    """metric: [n_concepts, n_layers, n_grid]. Returns per-layer bands and the best (highest-plateau) layer."""
    rows = []
    for li, layer in enumerate(layers):
        b = []
        for gi, N in enumerate(grid):
            vals = metric[:, li, :][:, gi]
            vals = vals[~np.isnan(vals)]
            if len(vals) >= MIN_CONCEPTS:
                b.append(
                    {
                        "N": int(N),
                        "median": float(np.median(vals)),
                        "p2_5": float(np.percentile(vals, 2.5)),
                        "p97_5": float(np.percentile(vals, 97.5)),
                        "n_concepts": int(len(vals)),
                    }
                )
        if b:
            rows.append({"layer": int(layer), "plateau": b[-1]["median"], "band": b})
    best = max(rows, key=lambda r: r["plateau"]) if rows else {}
    return rows, best


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    files = sorted(glob.glob(f"work/convcurve-{args.tag}/pair-*.npz"))
    if not files:
        raise SystemExit(f"no per-pair files under work/convcurve-{args.tag}/")
    first = np.load(files[0])
    layers = [int(x) for x in first["layers"]]
    grid = [int(x) for x in first["grid"]]
    aur = np.stack([np.load(f)["auroc"] for f in files])
    conv = np.stack([np.load(f)["conv"] for f in files])
    print(f"{args.tag}: {len(files)} pairs, {len(layers)} layers, grid {grid}", flush=True)

    aur_layers, aur_best = per_layer(aur, layers, grid)
    conv_layers, conv_best = per_layer(conv, layers, grid)
    summary = {
        "tag": args.tag,
        "grid": grid,
        "layers": layers,
        "n_pairs": len(files),
        "auroc": {"best_layer": aur_best.get("layer"), "best_band": aur_best.get("band"), "per_layer": aur_layers},
        "convergence": {
            "best_layer": conv_best.get("layer"),
            "best_band": conv_best.get("band"),
            "per_layer": conv_layers,
        },
    }
    out = Path("work/convcurve")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"convcurve-{args.tag}.json").write_text(json.dumps(summary))

    def show(name: str, best: dict) -> None:
        if not best:
            print(f"  {name}: no band reached {MIN_CONCEPTS} concepts")
            return
        print(f"\n=== {name}: best layer {best['layer']} ===")
        for e in best["band"]:
            print(
                f"   N={e['N']:>3}  median {e['median']:.3f}  [{e['p2_5']:.3f}, {e['p97_5']:.3f}]  ({e['n_concepts']} concepts)"
            )

    show("AUROC", aur_best)
    show("convergence", conv_best)
    print(f"\nwrote work/convcurve/convcurve-{args.tag}.json")


if __name__ == "__main__":
    main()
