from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import torch

from exp_attn_interp.config import INSTRUCT_MODEL, MARKERS_PRESENCE, N_BLOCKS_OF, PAIRS, READ_LAYER, vector_tag_for
from exp_attn_interp.vectors import load_diff_vectors

log = logging.getLogger("uniqmarkers")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--siblings", type=int, default=10)
    parser.add_argument("--layer", type=int, default=0, help="0 = half depth of --model (Qwen: 18)")
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--markers", type=Path, default=MARKERS_PRESENCE)
    parser.add_argument("--out", type=Path, default=Path("data/markers_unique.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(4)

    layer = args.layer or (READ_LAYER if args.model == INSTRUCT_MODEL else N_BLOCKS_OF[args.model] // 2)
    V = load_diff_vectors(device="cpu", tag=vector_tag_for(args.model))[layer]
    cos = V @ V.T
    cos.fill_diagonal_(-2.0)
    sib = cos.topk(args.siblings, dim=1).indices

    src = json.loads(args.markers.read_text())
    ids = {k: set(src["concept"][str(k)]) for k in range(PAIRS)}
    words = src.get("concept_words", {})
    unique: dict[str, list[int]] = {}
    sizes = []
    for k in range(PAIRS):
        shared: set[int] = set()
        for s in sib[k].tolist():
            shared |= ids[s]
        u = sorted(ids[k] - shared)
        unique[str(k)] = u
        sizes.append(len(u))
    sizes_t = torch.tensor(sizes, dtype=torch.float32)
    n_ge2 = int((sizes_t >= 2).sum())
    payload = {
        "source": "presence_gap_english_sibling_unique",
        "siblings": args.siblings,
        "layer": args.layer,
        "concept": unique,
        "antagonist": src.get("antagonist", {}),
        "concept_words": words,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload))
    log.info(
        "wrote %s: unique markers/concept median %.0f (p10 %.0f, p90 %.0f); %d/%d concepts keep >=2",
        args.out,
        sizes_t.median(),
        sizes_t.quantile(0.1),
        sizes_t.quantile(0.9),
        n_ge2,
        PAIRS,
    )


if __name__ == "__main__":
    main()
