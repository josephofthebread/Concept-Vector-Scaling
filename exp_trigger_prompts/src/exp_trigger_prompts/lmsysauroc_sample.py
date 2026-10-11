from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

log = logging.getLogger("lmsysauroc_sample")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path("data/lmsys_labels.parquet"))
    parser.add_argument("--out", type=Path, default=Path("work/lmsys-auroc/sample.json"))
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--user-words", type=int, nargs=2, default=(8, 250))
    parser.add_argument("--scan-cap", type=int, default=400_000)
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from datasets import load_dataset

    tab = pq.read_table(args.labels, columns=["conversation_id", "class_1_id"])
    label_of = dict(zip(tab["conversation_id"].to_pylist(), tab["class_1_id"].to_pylist()))
    log.info("loaded %d labels", len(label_of))

    umin, umax = args.user_words
    data = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    by_class: dict[int, list[dict]] = {}
    scanned = 0
    for row in data:
        scanned += 1
        if scanned > args.scan_cap:
            break
        if row.get("language") != "English":
            continue
        conv = row.get("conversation") or []
        if not conv or conv[0]["role"] != "user":
            continue
        cid = row["conversation_id"]
        klass = label_of.get(cid)
        if klass is None:
            continue
        klass = int(klass)
        if len(by_class.get(klass, [])) >= args.per_class:
            continue
        user = conv[0]["content"].strip()
        if not (umin <= len(user.split()) <= umax):
            continue
        by_class.setdefault(klass, []).append({"conversation_id": cid, "class_id": klass, "user": user})

    items = [it for lst in by_class.values() for it in lst]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(items, ensure_ascii=False))
    log.info(
        "scanned %d rows -> %d convs across %d classes (median %.1f/class) -> %s",
        scanned,
        len(items),
        len(by_class),
        sorted(len(v) for v in by_class.values())[len(by_class) // 2],
        args.out,
    )


if __name__ == "__main__":
    main()
