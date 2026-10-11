#! /usr/bin/env python
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors.torch import load_file

log = logging.getLogger("decompose")

VECTORS = Path("../exp_genvectors/work/vectors/Qwen_Qwen3-8B--assistant--semantic")
ROW = {11: 11, 14: 14, 18: 18, 22: 22, 25: 25}


def find(name: str) -> str:
    path = VECTORS / name
    if not path.is_file():
        raise SystemExit(f"{path} is missing; build the vectors with exp_genvectors first")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector", type=Path, required=True, help="unit direction as .npy")
    parser.add_argument(
        "--against",
        type=Path,
        action="append",
        default=[],
        help="other saved directions to report a cosine against; repeatable",
    )
    parser.add_argument("--layer", type=int, default=18, choices=sorted(ROW))
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rng = np.random.default_rng(args.seed)

    v = np.load(args.vector).astype(np.float64).flatten()
    v /= np.linalg.norm(v)

    block = load_file(find("diff.safetensors"))["diff"]
    basis = block[ROW[args.layer]].float().numpy().astype(np.float64)
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    rows = pq.read_table(find("pairs.parquet")).to_pylist()

    cosines = basis @ v
    order = np.argsort(-np.abs(cosines))[: args.top]
    _, singular, right = np.linalg.svd(basis, full_matrices=False)
    keep = right[singular > singular[0] * 1e-6]
    inside = keep.T @ (keep @ v)
    residual = v - inside
    share = float(inside @ inside)
    noise = rng.normal(size=(args.draws, basis.shape[1]))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    best = np.abs(noise @ basis.T).max(axis=1)
    _, singular, right = np.linalg.svd(basis, full_matrices=False)
    keep = right[singular > singular[0] * 1e-6]
    inside = noise @ keep.T
    reference = {
        "best_cosine_p50": float(np.quantile(best, 0.50)),
        "best_cosine_p95": float(np.quantile(best, 0.95)),
        "best_cosine_max": float(best.max()),
        "explained_p50": float(np.quantile((inside * inside).sum(axis=1), 0.50)),
        "explained_p95": float(np.quantile((inside * inside).sum(axis=1), 0.95)),
        "rank": int(keep.shape[0]),
    }

    report = {
        "vector": str(args.vector),
        "layer": args.layer,
        "explained_by_concept_span": share,
        "residual_norm": float(np.linalg.norm(residual)),
        "null": reference,
        "top": [
            {
                "pair": int(i),
                "cosine": float(cosines[i]),
                "concept": rows[i]["concept"],
                "antagonist": rows[i]["antagonist"],
                "class": rows[i]["class_name"],
            }
            for i in order
        ],
        "against": {},
    }

    for other in args.against:
        w = np.load(other).astype(np.float64).flatten()
        w /= np.linalg.norm(w)
        report["against"][Path(other).stem] = float(v @ w)

    log.info(f"{args.vector.name} at L{args.layer}")
    log.info(
        f"  concept span (rank {reference['rank']}) explains {share:.3f} of it; "
        f"a random direction gets {reference['explained_p50']:.3f} (p95 {reference['explained_p95']:.3f})"
    )
    log.info(
        f"  best |cos| to any of 1036: {abs(cosines[order[0]]):.3f}; "
        f"random gets {reference['best_cosine_p50']:.3f} (p95 {reference['best_cosine_p95']:.3f})"
    )
    for entry in report["top"][:8]:
        flag = "" if abs(entry["cosine"]) > reference["best_cosine_p95"] else "   (inside the null)"
        log.info(f"  {entry['cosine']:+.3f}  {entry['pair']:>4}  {entry['concept']} || {entry['antagonist']}{flag}")
    for name, value in report["against"].items():
        log.info(f"  cosine to {name}: {value:+.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
        log.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
