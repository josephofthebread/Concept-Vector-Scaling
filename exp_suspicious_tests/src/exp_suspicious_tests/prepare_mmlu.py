from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from exp_suspicious_tests.config import MMLU_PER_SUBJECT, MMLU_QUESTIONS, MMLU_SEED


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--extra", type=Path, default=Path("data/mcqa_extra.json"))
    parser.add_argument("--out", type=Path, default=Path("data/mcqa.parquet"))
    args = parser.parse_args()

    from datasets import load_dataset

    data = load_dataset("cais/mmlu", "all", split="test")
    by_subject: dict[str, list[dict]] = {}
    for row in data:
        by_subject.setdefault(row["subject"], []).append(row)

    rng = np.random.default_rng(MMLU_SEED)
    selected: list[dict] = []
    for subject in sorted(by_subject):
        rows = sorted(by_subject[subject], key=lambda r: r["question"])
        take = rng.choice(len(rows), size=min(MMLU_PER_SUBJECT, len(rows)), replace=False)
        for index in sorted(take.tolist()):
            row = rows[index]
            selected.append(
                {
                    "id": f"mmlu-{subject}-{index}",
                    "source": "mmlu",
                    "subject": subject,
                    "question": row["question"],
                    "options": list(row["choices"]),
                    "answer_index": int(row["answer"]),
                }
            )
    order = rng.permutation(len(selected))[:MMLU_QUESTIONS]
    selected = [selected[index] for index in sorted(order.tolist())]

    crafted = json.loads(args.extra.read_text())
    for item in crafted:
        assert len(item["options"]) == 4 and item["source"] in ("fictional", "twin", "postcutoff", "unknowable")
    items = selected + crafted
    table = pa.Table.from_pylist(items)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out)
    counts = {
        source: sum(1 for item in items if item["source"] == source)
        for source in ("mmlu", "fictional", "twin", "postcutoff", "unknowable")
    }
    print(f"wrote {args.out}: {len(items)} items {counts}")


if __name__ == "__main__":
    main()
