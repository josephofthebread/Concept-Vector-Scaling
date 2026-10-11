from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("select")
    parser = ArgumentParser()
    parser.add_argument("--work", type=Path, default=Path("work"))
    parser.add_argument("--features", type=Path, default=Path("data/features.json"))
    args = parser.parse_args()

    meta_files = sorted(args.work.glob("ids-*.json"))
    if not meta_files:
        raise SystemExit(f"no ids-*.json in {args.work}")
    id_lists = [json.loads(m.read_text())["ids"] for m in meta_files]
    shards = [json.loads(m.read_text())["shard"] for m in meta_files]
    k = json.loads(meta_files[0].read_text())["topk"]
    blobs = [load_file(str(args.work / f"cand-{s}.safetensors")) for s in shards]
    features = json.loads(args.features.read_text())["features"]
    pairs = len(features)
    log.info("%d shards, %d features, top-%d per pole", len(shards), pairs, k)

    out = []
    for p, feat in enumerate(features):
        row: dict = {
            "pair_id": p,
            "concept": feat["concept"],
            "antagonist": feat["antagonist"],
            "class": feat["class_name"],
        }
        for pole in ("con", "ant"):
            score = np.concatenate([b[f"{pole}_score"][:, p] for b in blobs])
            conv = np.concatenate([b[f"{pole}_conv"][:, p] for b in blobs])
            layer = np.concatenate([b[f"{pole}_layer"][:, p] for b in blobs])
            owner = np.repeat(np.arange(len(shards)), k)
            top = np.argsort(-score)[:k]
            row[pole] = [
                {
                    "conversation_id": id_lists[owner[t]][int(conv[t])],
                    "score": round(float(score[t]), 4),
                    "layer": int(layer[t]),
                }
                for t in top
                if np.isfinite(score[t]) and conv[t] >= 0
            ]
        out.append(row)

    (args.work / "candidates.json").write_text(
        json.dumps({"topk": k, "n_features": pairs, "features": out}, ensure_ascii=False)
    )
    log.info("wrote candidates.json (%d features, concept+antagonist top-%d each)", pairs, k)


if __name__ == "__main__":
    main()
