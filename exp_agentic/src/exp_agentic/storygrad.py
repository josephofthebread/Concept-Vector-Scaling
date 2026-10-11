#! /usr/bin/env python


from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("storygrad")

PREFIX = "<|im_start|>user\nWrite a short story.<|im_end|>\n<|im_start|>assistant\n"
CHUNK = 512


def logprob(model: Any, ids: Any, skip: int) -> Any:
    """Log-probability the model assigns to a story, past the shared prefix."""
    import torch

    hidden = model.model(input_ids=ids).last_hidden_state[0]
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(max(skip - 1, 0), hidden.shape[0] - 1, CHUNK):
        stop = min(start + CHUNK, hidden.shape[0] - 1)
        logits = model.lm_head(hidden[start:stop]).float()
        chosen = ids[0, start + 1 : stop + 1]
        total = total + torch.log_softmax(logits, dim=-1).gather(1, chosen[:, None])[:, 0].sum()
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("storygrad"))
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--per-pole", type=int, default=500, help="stories sampled per pole")
    parser.add_argument(
        "--keys-file",
        type=Path,
        default=Path("pairs-32-keys.json"),
        help="JSON map of pair id -> [concept, antagonist] identifying strings",
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import torch
    from datasets import load_dataset

    from exp_agentic.bipo import inject
    from exp_agentic.model import load

    model, tokenizer = load(device=args.device, dtype=args.dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    width = model.config.hidden_size
    vector = torch.zeros(width, device=args.device, dtype=torch.float32, requires_grad=True)
    inject(model, args.layer, vector)

    prefix_ids = tokenizer(PREFIX, add_special_tokens=False)["input_ids"]
    skip = len(prefix_ids)
    log.info(f"rendered prefix is {skip} tokens; scoring begins after it")

    log.info("loading AntonKorznikov/feature_stories")
    data = load_dataset("AntonKorznikov/feature_stories", split="train")
    log.info(f"{len(data):,} rows, columns {data.column_names}")

    required = ("concept", "antagonist", "concept_text", "antagonist_text")
    if any(name not in data.column_names for name in required):
        raise SystemExit(f"expected {required} among {data.column_names}")

    keys = json.loads(args.keys_file.read_text())
    lookup = {(c.strip(), a.strip()): int(i) for i, (c, a) in keys.items()}
    log.info(f"{len(lookup)} requested pairs, keyed by concept/antagonist text")

    pairs = max(lookup.values()) + 1
    sums = np.zeros((pairs, 2, width), dtype=np.float64)
    counts = np.zeros((pairs, 2), dtype=np.int64)
    rng = np.random.default_rng(args.seed)

    by_pair: dict[int, list[int]] = {}
    for index, (concept, antagonist) in enumerate(zip(data["concept"], data["antagonist"])):
        found = lookup.get((str(concept).strip(), str(antagonist).strip()))
        if found is not None:
            by_pair.setdefault(found, []).append(index)

    missing = set(lookup.values()) - set(by_pair)
    if missing:
        log.warning(f"{len(missing)} requested pairs matched NO rows: {sorted(missing)[:8]}")
    if not by_pair:
        raise SystemExit("no dataset rows matched any requested pair -- the key strings do not align")
    log.info(
        f"matched {len(by_pair)} pairs, "
        f"{min(len(v) for v in by_pair.values())}-{max(len(v) for v in by_pair.values())} rows each"
    )

    selected = sorted(by_pair)
    mine = [p for i, p in enumerate(selected) if i % args.shards == args.shard]

    wanted: dict[int, np.ndarray] = {}
    for pair in mine:
        picks = np.array(by_pair[pair])
        if len(picks) > args.per_pole:
            picks = rng.choice(picks, size=args.per_pole, replace=False)
        wanted[pair] = picks

    log.info(
        f"shard {args.shard}/{args.shards}: {len(mine)} pairs, "
        f"{sum(len(v) for v in wanted.values()):,} rows -> "
        f"{2 * sum(len(v) for v in wanted.values()):,} stories"
    )

    seen = 0
    for pair in sorted(wanted):
        for index in wanted[pair]:
            row = data[int(index)]
            for pole, field in ((0, "concept_text"), (1, "antagonist_text")):
                text = row[field]
                if not text:
                    continue
                body = tokenizer(text, add_special_tokens=False, truncation=True, max_length=args.max_tokens)[
                    "input_ids"
                ]
                if len(body) < 16:
                    continue
                ids = torch.tensor([prefix_ids + body], device=args.device)
                if vector.grad is not None:
                    vector.grad = None
                logprob(model, ids, skip).backward()
                sums[pair, pole] += vector.grad.detach().float().cpu().numpy()
                counts[pair, pole] += 1
                seen += 1
            if seen % 500 == 0 and seen:
                log.info(f"{seen:,} stories accumulated over {int((counts > 0).all(axis=1).sum())} pairs")

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "storygrad.npz", sums=sums, counts=counts, layer=np.array(args.layer), shard=np.array(args.shard)
    )
    filled = int((counts > 0).all(axis=1).sum())
    log.info(f"shard {args.shard} complete: {seen:,} stories over {filled} pairs with both poles")
    (args.out / "storygrad.json").write_text(
        json.dumps(
            {
                "layer": args.layer,
                "shard": args.shard,
                "shards": args.shards,
                "stories": seen,
                "pairs_complete": filled,
                "pairs": pairs,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
