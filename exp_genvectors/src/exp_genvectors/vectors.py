from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np

from exp_genvectors.artifacts import atomic_safetensors, canonical_json, read_capture
from exp_genvectors.vecmath import contrasts, split_reliability, whiten, within_covariance

log = logging.getLogger("vectors")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--shrinkage", type=float, default=0.05)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = args.inputs
    if not paths:
        raise ValueError("no capture shards supplied")
    manifest: dict[str, Any] | None = None
    totals: dict[str, np.ndarray] = {}
    seen: set[int] = set()
    for path in sorted(paths):
        current, tensors = read_capture(path)
        if manifest is None:
            manifest = current
            totals = {
                name: tensor.astype(np.float64 if np.issubdtype(tensor.dtype, np.floating) else np.int64)
                for name, tensor in tensors.items()
            }
        else:
            if current["merge_key"] != manifest["merge_key"]:
                raise ValueError(f"incompatible capture shard {path}")
            if tensors.keys() != totals.keys():
                raise ValueError(f"tensor set differs in {path}")
            for name, tensor in tensors.items():
                totals[name] += tensor
        if int(current["shard"]) in seen:
            raise ValueError(f"duplicate shard {current['shard']}: {path}")
        seen.add(int(current["shard"]))
    assert manifest is not None
    if seen != set(range(int(manifest["shards"]))):
        raise ValueError(f"capture incomplete: have shards {sorted(seen)} of {manifest['shards']}")
    pairs_meta = json.loads(args.ontology.read_text())["pairs"]
    pairs_meta.sort(key=lambda pair: int(pair["pair_id"]))
    sums, counts, corpus_sum = totals["sums"], totals["counts"], totals["corpus_sum"]
    layers = int(manifest["layers"])
    hidden = int(manifest["hidden_size"])
    pairs = sums.shape[1]
    if len(pairs_meta) != pairs:
        raise ValueError(f"ontology has {len(pairs_meta)} pairs, capture has {pairs}")

    raw = contrasts(sums, counts, corpus_sum)
    whitened = {name: np.empty((layers, pairs, hidden), dtype=np.float32) for name in raw}
    spectra = np.empty((layers, hidden))
    reliability_raw, reliability_white = [], []
    reliability_raw_250, reliability_white_250 = [], []
    for layer in range(layers):
        covariance = within_covariance(
            totals["moments"][layer].sum(axis=0),
            sums[layer].reshape(pairs * 2, -1, hidden).sum(axis=1),
            counts.reshape(pairs * 2, -1).sum(axis=1),
        )
        spectra[layer] = np.linalg.eigvalsh(covariance)[::-1]
        stacked = np.concatenate([raw[name][layer] for name in raw], axis=0)
        solved = whiten(stacked, covariance, args.shrinkage)
        for position, name in enumerate(raw):
            whitened[name][layer] = solved[position * pairs : (position + 1) * pairs].astype(np.float32)
        reliability = split_reliability(sums, counts, totals["moments"], layer=layer, shrinkage=args.shrinkage)
        reliability_raw.append(reliability["raw"])
        reliability_white.append(reliability["whitened"])
        quarter = split_reliability(sums, counts, totals["moments"], layer=layer, shrinkage=args.shrinkage, size=1)
        reliability_raw_250.append(quarter["raw"])
        reliability_white_250.append(quarter["whitened"])
        log.info(
            "layer %s: median split-half cosine raw %.4f, whitened %.4f",
            layer,
            float(np.median(reliability["raw"])),
            float(np.median(reliability["whitened"])),
        )

    output_manifest = {
        "schema_version": 2,
        "model": manifest["model"],
        "presentation": manifest["presentation"],
        "dataset": manifest["dataset"],
        "layers": layers,
        "hidden_size": hidden,
        "pairs": pairs,
        "shrinkage": args.shrinkage,
        "capture_merge_key": manifest["merge_key"],
        "axes": {"raw": ["layer", "pair", "hidden"], "whitened": ["layer", "pair", "hidden"]},
    }
    metadata = {"manifest": canonical_json(output_manifest)}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, tensor in raw.items():
        atomic_safetensors({name: tensor.astype(np.float32)}, args.out / f"{name}.safetensors", metadata)
        atomic_safetensors({f"whitened_{name}": whitened[name]}, args.out / f"whitened_{name}.safetensors", metadata)
    diagnostics = {
        "eigenvalues": spectra,
        "reliability_raw": np.stack(reliability_raw).astype(np.float32),
        "reliability_whitened": np.stack(reliability_white).astype(np.float32),
        "reliability_raw_250": np.stack(reliability_raw_250).astype(np.float32),
        "reliability_whitened_250": np.stack(reliability_white_250).astype(np.float32),
    }
    atomic_safetensors(diagnostics, args.out / "diagnostics.safetensors", metadata)

    import pyarrow as pa
    import pyarrow.parquet as pq

    pair_rows = counts.sum(axis=(1, 2))
    table: dict[str, Any] = {
        "pair_id": np.arange(pairs, dtype=np.int32),
        "concept": [pair["concept"] for pair in pairs_meta],
        "antagonist": [pair["antagonist"] for pair in pairs_meta],
        "class_name": [pair["class_name"] for pair in pairs_meta],
        "rows": pair_rows // 2,
        "label_hits": totals["label_hits"].sum(axis=(1, 2)),
    }
    for layer in range(layers):
        table[f"L{layer:02d}_diff_norm"] = np.linalg.norm(raw["diff"][layer], axis=1)
        table[f"L{layer:02d}_directional_agreement"] = (
            2 * np.linalg.norm(totals["unit_diff_sum"][layer].sum(axis=1), axis=1) / np.maximum(pair_rows, 1)
        )
    pq.write_table(pa.table(table), args.out / "pairs.parquet", compression="zstd")
    (args.out / "manifest.json").write_text(json.dumps(output_manifest, indent=2))


if __name__ == "__main__":
    main()
