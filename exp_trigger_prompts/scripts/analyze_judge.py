from __future__ import annotations

import collections
import glob
import json
from argparse import ArgumentParser
from pathlib import Path


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("work/judge"))
    ap.add_argument("--glob", default="scores-shard-*.jsonl")
    ap.add_argument("--out-dir", type=Path, default=Path("work/judge"))
    args = ap.parse_args()

    files = sorted(glob.glob(str(args.in_dir / args.glob)))
    if not files:
        raise SystemExit(f"no score shards under {args.in_dir}/{args.glob}")

    cell: dict[tuple[int, int, float], list[float]] = collections.defaultdict(list)
    n_rows = 0
    neg = 0
    for fp in files:
        with open(fp) as fh:
            for line in fh:
                r = json.loads(line)
                c, i, f = r["concept"], r["instruct"], r["fluency"]
                if min(c, i, f) < 0:
                    neg += 1
                    continue
                cell[(r["pair_id"], r["layer"], r["alpha"])].append(
                    0.0 if c <= 0 or i <= 0 or f <= 0 else 3.0 / (1.0 / c + 1.0 / i + 1.0 / f)
                )
                n_rows += 1

    per_cell = args.out_dir / "steerscore-cells.jsonl"
    with per_cell.open("w") as sink:
        for (pid, layer, alpha), vals in sorted(cell.items()):
            sink.write(
                json.dumps(
                    {
                        "pair_id": pid,
                        "layer": layer,
                        "alpha": alpha,
                        "score": sum(vals) / len(vals),
                        "n": len(vals),
                    }
                )
                + "\n"
            )

    grid: dict[tuple[int, float], list[float]] = collections.defaultdict(list)
    for (pid, layer, alpha), vals in cell.items():
        grid[(layer, alpha)].append(sum(vals) / len(vals))
    layers = sorted({l for l, _ in grid})
    alphas = sorted({a for _, a in grid})
    grid_mean = {k: sum(v) / len(v) for k, v in grid.items()}

    grid_out = args.out_dir / "steerscore-grid.json"
    grid_out.write_text(
        json.dumps(
            {
                "layers": layers,
                "alphas": alphas,
                "mean": {f"L{l}|a{a:g}": grid_mean[(l, a)] for (l, a) in grid_mean},
                "concepts": len({pid for pid, _, _ in cell}),
                "rows": n_rows,
                "neg_dropped": neg,
            },
            indent=2,
        )
    )

    print(f"rows={n_rows}  concepts={len({pid for pid, _, _ in cell})}  cells={len(cell)}  neg_dropped={neg}")
    print("\nsteering score (harmonic mean), averaged over all concepts")
    print("layer \\ alpha  " + "  ".join(f"{a:>+6.2g}" for a in alphas))
    best = (-1.0, None)
    for l in layers:
        row = "  ".join(f"{grid_mean[(l, a)]:6.3f}" for a in alphas)
        print(f"    L{l:<8d}  {row}")
    for k, v in grid_mean.items():
        if v > best[0]:
            best = (v, k)
    by_layer = {l: sum(grid_mean[(l, a)] for a in alphas) / len(alphas) for l in layers}
    by_alpha = {a: sum(grid_mean[(l, a)] for l in layers) / len(layers) for a in alphas}
    print("\nmean by layer :", "  ".join(f"L{l}={by_layer[l]:.3f}" for l in layers))
    print("mean by alpha :", "  ".join(f"{a:+g}={by_alpha[a]:.3f}" for a in alphas))
    bl, ba = best[1]
    print(f"\nbest cell     : L{bl} alpha={ba:+g}  score={best[0]:.3f}")
    print(f"best layer    : L{max(by_layer, key=by_layer.get)}")
    print(f"best |alpha|  : {max(by_alpha, key=by_alpha.get):+g}")
    print(f"\nwrote {per_cell}\nwrote {grid_out}")


if __name__ == "__main__":
    main()
