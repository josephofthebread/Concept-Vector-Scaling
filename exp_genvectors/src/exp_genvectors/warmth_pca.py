from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from safetensors import safe_open

log = logging.getLogger("warmth-pca")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--src", required=True, type=Path, help="perstory output directory")
    parser.add_argument("--out", required=True, type=Path, help="destination directory")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="unit-normalize each story vector before PCA, matching the PR convention",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    paths = sorted(args.src.glob("pair-*.safetensors"))
    if not paths:
        raise SystemExit(f"no per-story artifacts in {args.src}")

    blocks, rows = [], []
    for path in paths:
        with safe_open(path, framework="np") as handle:
            manifest = json.loads(handle.metadata()["manifest"])
            vectors = handle.get_slice("diff")[:, args.layer, :]
        vectors = np.asarray(vectors, dtype=np.float32)
        if args.normalize:
            vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
        blocks.append(vectors)
        for position, row in enumerate(manifest["rows"]):
            rows.append(
                {
                    "pair_id": manifest["pair_id"],
                    "concept": manifest["concept"],
                    "antagonist": manifest["antagonist"],
                    "row_id": row["row_id"],
                    "genre_id": row["genre_id"],
                    "language_id": row["language_id"],
                    "generator_id": row["generator_id"],
                    "bucket": row["bucket"],
                    "position": position,
                }
            )
        log.info("%s: %s stories, %s", path.name, vectors.shape[0], manifest["concept"])

    matrix = np.concatenate(blocks, axis=0)
    center = matrix.mean(axis=0)
    centered = matrix - center
    _, singular, basis = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2
    ratio = variance / variance.sum()
    coords = centered @ basis[:2].T

    args.out.mkdir(parents=True, exist_ok=True)
    header = "pair_id,concept,antagonist,row_id,genre_id,language_id,generator_id,bucket,pc1,pc2"
    lines = [header]
    for record, (x, y) in zip(rows, coords, strict=True):
        lines.append(
            f'{record["pair_id"]},"{record["concept"]}","{record["antagonist"]}",'
            f"{record['row_id']},{record['genre_id']},{record['language_id']},"
            f"{record['generator_id']},{record['bucket']},{x:.6f},{y:.6f}"
        )
    (args.out / f"points-L{args.layer}.csv").write_text("\n".join(lines) + "\n")

    np.savez_compressed(
        args.out / f"basis-L{args.layer}.npz",
        basis=basis[:16].astype(np.float32),
        center=center.astype(np.float32),
        explained_variance_ratio=ratio[:64].astype(np.float64),
    )
    summary = {
        "source": str(args.src),
        "layer": args.layer,
        "normalized": bool(args.normalize),
        "stories": int(matrix.shape[0]),
        "hidden": int(matrix.shape[1]),
        "concepts": sorted({(r["pair_id"], r["concept"]) for r in rows}),
        "pc1_variance": float(ratio[0]),
        "pc2_variance": float(ratio[1]),
        "participation_ratio": float(variance.sum() ** 2 / (variance**2).sum()),
    }
    (args.out / f"summary-L{args.layer}.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    log.info(
        "PC1 %.1f%%, PC2 %.1f%%, PR %.1f over %s stories",
        ratio[0] * 100,
        ratio[1] * 100,
        summary["participation_ratio"],
        matrix.shape[0],
    )


if __name__ == "__main__":
    main()
