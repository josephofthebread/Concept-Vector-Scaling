#! /usr/bin/env python
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors.numpy import load_file

log = logging.getLogger("families")

VECTORS = Path("../exp_genvectors/work/vectors/Qwen_Qwen3-8B--assistant--semantic")
LAYERS = (11, 14, 18, 22, 25)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vector", type=Path, action="append", required=True, help="extracted direction as .npy; repeatable"
    )
    parser.add_argument("--readouts", type=Path, default=Path("readouts.safetensors"))
    parser.add_argument("--layer", type=int, default=18, choices=LAYERS)
    parser.add_argument("--out", type=Path, default=Path("analysis/families.json"))
    parser.add_argument("--residuals", type=Path, default=Path("bipo"))
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rng = np.random.default_rng(args.seed)

    held = load_file(args.readouts)
    stacked = held["readouts"]
    methods = [
        "diff",
        "concept_centered",
        "antagonist_centered",
        "whitened_diff",
        "whitened_concept_centered",
        "whitened_antagonist_centered",
    ]
    if stacked.shape[0] != len(methods):
        raise SystemExit(f"expected {len(methods)} methods in {args.readouts}, found {stacked.shape[0]}")
    position = LAYERS.index(args.layer)
    rows = pq.read_table(VECTORS / "pairs.parquet").to_pylist()

    report: dict = {"layer": args.layer, "methods": methods, "vectors": {}}
    for path in args.vector:
        v = np.load(path).astype(np.float64).flatten()
        v /= np.linalg.norm(v)
        log.info(f"=== {path.name} at L{args.layer} ===")
        entry: dict = {}
        for index, method in enumerate(methods):
            basis = stacked[index, position].astype(np.float64)
            unit = basis / np.maximum(np.linalg.norm(basis, axis=1, keepdims=True), 1e-12)
            _, singular, right = np.linalg.svd(unit, full_matrices=False)
            span = right[singular > singular[0] * 1e-6]

            cosines = unit @ v
            inside = span.T @ (span @ v)
            residual = v - inside

            noise = rng.normal(size=(args.draws, unit.shape[1]))
            noise /= np.linalg.norm(noise, axis=1, keepdims=True)
            null_best = np.abs(noise @ unit.T).max(axis=1)
            null_span = ((noise @ span.T) ** 2).sum(axis=1)

            got = {
                "rank": int(span.shape[0]),
                "best_cosine": float(np.abs(cosines).max()),
                "best_cosine_null_p95": float(np.quantile(null_best, 0.95)),
                "explained": float(inside @ inside),
                "explained_null_p95": float(np.quantile(null_span, 0.95)),
            }
            order = np.argsort(-np.abs(cosines))[: args.top]
            got["top"] = [
                {
                    "pair": int(j),
                    "cosine": float(cosines[j]),
                    "concept": rows[j]["concept"],
                    "antagonist": rows[j]["antagonist"],
                    "class": rows[j]["class_name"],
                }
                for j in order
            ]
            entry[method] = got

            norm = np.linalg.norm(residual)
            if norm > 1e-9:
                args.residuals.mkdir(parents=True, exist_ok=True)
                target = args.residuals / f"{path.stem}-residual-{method}-L{args.layer}.npy"
                np.save(target, (residual / norm).astype(np.float32))

            flag = "" if got["best_cosine"] > got["best_cosine_null_p95"] else "  (inside null)"
            log.info(
                f"  {method:<32} rank {got['rank']:>4}  best|cos| {got['best_cosine']:.3f} "
                f"(null {got['best_cosine_null_p95']:.3f}){flag}  span explains "
                f"{got['explained']:.3f} (null {got['explained_null_p95']:.3f})"
            )
            for row in got["top"][:3]:
                log.info(f"      {row['cosine']:+.3f}  {row['pair']:>4}  {row['concept']} || {row['antagonist']}")
        report["vectors"][path.stem] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    log.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
