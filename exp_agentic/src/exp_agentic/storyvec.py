#! /usr/bin/env python


from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors.numpy import load_file

log = logging.getLogger("storyvec")

VECTORS = Path("../exp_genvectors/work/vectors/Qwen_Qwen3-8B--assistant--semantic")
LAYERS = (11, 14, 18, 22, 25)
FAMILIES = (
    "diff",
    "concept_centered",
    "antagonist_centered",
    "whitened_diff",
    "whitened_concept_centered",
    "whitened_antagonist_centered",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("storygrad"))
    parser.add_argument("--readouts", type=Path, default=Path("../readouts.safetensors"))
    parser.add_argument("--blocks", type=Path, default=Path("blocks-32.json"))
    parser.add_argument("--layer", type=int, default=18, choices=LAYERS)
    parser.add_argument("--out", type=Path, default=Path("analysis/storyvec.json"))
    parser.add_argument("--vectors-out", type=Path, default=Path("storygrad/gradvec-L18.npy"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    shards = sorted(args.dir.glob("shard-*/storygrad.npz")) or sorted(args.dir.glob("storygrad*.npz"))
    sums = counts = None
    for path in [Path(p) for p in shards]:
        held = np.load(path)
        s, c = held["sums"], held["counts"]
        if s.shape[-1] < 64:
            continue
        sums = s.copy() if sums is None else sums + s
        counts = c.copy() if counts is None else counts + c
    if sums is None:
        raise SystemExit("no usable storygrad shards")
    log.info(
        f"merged {len(shards)} shards: {int(counts.sum()):,} stories, "
        f"{int((counts > 0).all(axis=1).sum())} pairs with both poles"
    )

    live = (counts > 0).all(axis=1)
    ids = np.flatnonzero(live)
    means = sums / np.maximum(counts, 1)[:, :, None]
    directions = means[:, 0] - means[:, 1]
    unit = directions / np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)

    rows = pq.read_table(VECTORS / "pairs.parquet").to_pylist()
    stacked = load_file(args.readouts)["readouts"]
    position = LAYERS.index(args.layer)

    report: dict = {"layer": args.layer, "pairs": ids.tolist(), "stories": int(counts.sum()), "families": {}}

    for index, family in enumerate(FAMILIES):
        basis = stacked[index, position].astype(np.float64)
        basis = basis / np.maximum(np.linalg.norm(basis, axis=1, keepdims=True), 1e-12)
        matched = np.array([float(unit[p] @ basis[p]) for p in ids])
        crossed = np.array([float(unit[p] @ basis[q]) for p in ids for q in ids if p != q])
        report["families"][family] = {
            "matched_median": float(np.median(matched)),
            "matched_mean": float(matched.mean()),
            "crossed_abs_median": float(np.median(np.abs(crossed))),
            "matched": {int(p): float(c) for p, c in zip(ids, matched)},
        }
        log.info(
            f"{family:<32} own-pair cosine median {np.median(matched):+.3f}, "
            f"|cross-pair| median {np.median(np.abs(crossed)):.3f}"
        )

    if args.blocks.exists():
        blocks = json.loads(args.blocks.read_text())
        member = {p: name for name, ps in blocks.items() for p in ps}
        within, across = [], []
        for a in ids:
            for b in ids:
                if a >= b:
                    continue
                value = abs(float(unit[a] @ unit[b]))
                (within if member.get(int(a)) == member.get(int(b)) else across).append(value)
        report["blocks"] = {
            "within_median": float(np.median(within)) if within else None,
            "across_median": float(np.median(across)) if across else None,
        }
        log.info(
            f"block structure: |cos| within {np.median(within):.3f} vs across "
            f"{np.median(across):.3f}  ({'structure present' if np.median(within) > np.median(across) else 'NO STRUCTURE'})"
        )

    args.vectors_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.vectors_out, unit.astype(np.float32))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    log.info(f"wrote {args.vectors_out} and {args.out}")
    for p in ids[:8]:
        log.info(f"  {p:>4} {rows[p]['concept']} || {rows[p]['antagonist']}")


if __name__ == "__main__":
    main()
