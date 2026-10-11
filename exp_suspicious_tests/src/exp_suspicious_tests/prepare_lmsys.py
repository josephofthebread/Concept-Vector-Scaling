from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from exp_suspicious_tests.config import EXP4_ITEMS, EXP4_MAX_TOKENS, EXP4_MIN_TOKENS, EXP4_SEED, INSTRUCT_MODEL


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path("data/lmsys_labels.parquet"))
    parser.add_argument("--out", type=Path, default=Path("data/lmsys_items.parquet"))
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    labels = pq.read_table(args.labels, columns=["conversation_id", "class_1_id"])
    class_of = dict(zip(labels["conversation_id"].to_pylist(), labels["class_1_id"].to_pylist()))
    tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_MODEL, use_fast=True)

    data = load_dataset("lmsys/lmsys-chat-1m", split="train")
    by_class: dict[int, list[tuple[str, str]]] = {}
    for row in data:
        if row["language"] != "English":
            continue
        conversation = row["conversation"]
        if not conversation or conversation[0]["role"] != "user":
            continue
        klass = class_of.get(row["conversation_id"])
        if klass is None:
            continue
        text = conversation[0]["content"].strip()
        words = len(text.split())
        if words < 8 or words > 2000:
            continue
        by_class.setdefault(int(klass), []).append((row["conversation_id"], text))

    rng = np.random.default_rng(EXP4_SEED)
    for klass in by_class:
        by_class[klass].sort(key=lambda pair: pair[0])
        rng.shuffle(by_class[klass])

    items: list[dict] = []
    cursors = {klass: 0 for klass in by_class}
    classes = sorted(by_class)
    while len(items) < EXP4_ITEMS and classes:
        for klass in list(classes):
            pool = by_class[klass]
            while cursors[klass] < len(pool):
                conversation_id, text = pool[cursors[klass]]
                cursors[klass] += 1
                if EXP4_MIN_TOKENS <= len(tokenizer(text, add_special_tokens=False)["input_ids"]) <= EXP4_MAX_TOKENS:
                    items.append({"conversation_id": conversation_id, "class_id": klass, "text": text})
                    break
            else:
                classes.remove(klass)
            if len(items) >= EXP4_ITEMS:
                break

    if len(items) < EXP4_ITEMS:
        raise SystemExit(f"only {len(items)} items survived the filters; need {EXP4_ITEMS}")
    order = rng.permutation(len(items))
    items = [items[i] for i in order]
    for item_id, item in enumerate(items):
        item["item_id"] = item_id
        item["chunk"] = item_id // 64
    pq.write_table(pa.Table.from_pylist(items), args.out)
    counts = len(set(item["class_id"] for item in items))
    print(f"wrote {args.out}: {len(items)} items across {counts} classes, chunks 0..{items[-1]['chunk']}")


if __name__ == "__main__":
    main()
