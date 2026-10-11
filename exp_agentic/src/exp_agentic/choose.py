#! /usr/bin/env python
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

log = logging.getLogger("choose")

VECTORS = Path("../exp_genvectors/work/vectors/Qwen_Qwen3-8B--assistant--semantic")
ANCHORS = (376, 316, 463, 258, 611, 129, 548, 272, 519, 1014, 251, 200, 181, 906, 584, 209, 270, 296)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("pairs-128.json"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rng = np.random.default_rng(args.seed)

    rows = pq.read_table(VECTORS / "pairs.parquet").to_pylist()
    by_class: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_class[row["class_name"]].append(index)

    picked = {p for p in ANCHORS if p < len(rows)}
    log.info(f"{len(rows)} pairs across {len(by_class)} classes; {len(picked)} anchors forced in")

    classes = sorted(by_class, key=lambda c: -len(by_class[c]))
    remaining = args.count - len(picked)
    for name in classes:
        if remaining <= 0:
            break
        pool = [i for i in by_class[name] if i not in picked]
        if not pool:
            continue
        share = max(1, round(args.count * len(by_class[name]) / len(rows)))
        take = min(share, len(pool), remaining)
        for index in rng.choice(pool, size=take, replace=False):
            picked.add(int(index))
        remaining = args.count - len(picked)

    pool = [i for i in range(len(rows)) if i not in picked]
    while len(picked) < args.count and pool:
        picked.add(int(rng.choice(pool)))
        pool = [i for i in range(len(rows)) if i not in picked]

    final = sorted(picked)[: args.count]
    covered = {rows[i]["class_name"] for i in final}
    args.out.write_text(json.dumps(final))
    log.info(f"wrote {args.out}: {len(final)} pairs covering {len(covered)}/{len(by_class)} classes")
    for index in final[:10]:
        log.info(f"  {index:>4} [{rows[index]['class_name']}] {rows[index]['concept']} || {rows[index]['antagonist']}")


if __name__ == "__main__":
    main()
