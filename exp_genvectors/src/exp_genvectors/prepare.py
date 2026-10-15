from __future__ import annotations

import json
import logging
import re
import unicodedata
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from exp_genvectors.config import BUCKETS, GENRES, LANGUAGES, SOURCE_DATASET, VARIANTS

log = logging.getLogger("prepare")

ROWS_PER_PAIR = 1000
BUCKET_SEED = 20260814
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")


def bucket_assignment(pair_id: int, rows: int = ROWS_PER_PAIR) -> np.ndarray:
    """Seeded permutation with exactly rows/BUCKETS entries per bucket."""
    if rows % BUCKETS:
        raise ValueError(f"{rows} rows cannot split evenly into {BUCKETS} buckets")
    assignment = np.repeat(np.arange(BUCKETS), rows // BUCKETS)
    np.random.default_rng(np.random.SeedSequence([BUCKET_SEED, pair_id])).shuffle(assignment)
    return assignment


def no_punctuation(text: str) -> str:
    kept = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    return re.sub(r"[ \t]+", " ", kept).strip()


def hard_newlines(text: str, width: int = 50) -> str:
    return "\n".join(text[start : start + width] for start in range(0, len(text), width))


def markdown_bullets(text: str) -> str:
    sentences = [part.strip() for part in SENTENCE_BOUNDARY.split(text) if part.strip()]
    return "\n".join(f"- {sentence}" for sentence in sentences)


TRANSFORMS: tuple[tuple[str, str, str, Callable[[str], str]], ...] = (
    ("caps", "ALL CAPS", "normal case", str.upper),
    ("lowercase", "all lowercase", "normal case", str.lower),
    ("no_punctuation", "no punctuation", "normal punctuation", no_punctuation),
    ("hard_newlines_50", "newline every 50 characters", "flowing prose", hard_newlines),
    ("markdown_bullets", "markdown bullets", "prose paragraphs", markdown_bullets),
    ("identity_trailing_space", "story plus one trailing space", "story", lambda text: text + " "),
)


def write_parquet(records: list[dict[str, Any]], path: Path) -> None:
    partial = path.with_suffix(".partial")
    pq.write_table(pa.Table.from_pylist(records), partial, compression="zstd")
    partial.replace(path)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--source", default=SOURCE_DATASET)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from datasets import load_dataset

    data = load_dataset(args.source, split="train")
    ontology = json.loads(Path(hf_hub_download(args.source, "ontology.json", repo_type="dataset")).read_text())
    rows = [
        {
            "class_name": group["name"],
            "concept": pair["concept"],
            "antagonist": pair["antagonist"],
            "narrative_guidance": pair.get("narrative_guidance", ""),
        }
        for group in ontology["classes"]
        for pair in group["pairs"]
    ]
    rows.sort(key=lambda row: (row["concept"], row["antagonist"]))
    if len({(row["concept"], row["antagonist"]) for row in rows}) != len(rows):
        raise ValueError("ontology contains duplicate concept/antagonist pairs")
    pairs = [{"pair_id": index, **row} for index, row in enumerate(rows)]
    log.info("source: %s rows, %s pairs", len(data), len(pairs))

    source = list(data)
    pair_ids = {(row["concept"], row["antagonist"]): row["pair_id"] for row in pairs}
    generators = sorted({str(row["model"]) for row in source})
    if len(generators) != 2:
        raise ValueError(f"expected exactly two generator models, found {generators}")
    buckets = {row["pair_id"]: bucket_assignment(row["pair_id"]) for row in pairs}
    ordinals: Counter[int] = Counter()
    records = []
    for index, row in enumerate(source):
        key = (row["concept"], row["antagonist"])
        if key not in pair_ids:
            raise ValueError(f"source row {index} has pair not in ontology: {key}")
        if not row["concept_text"] or not row["antagonist_text"]:
            raise ValueError(f"source row {index} has an empty pole text")
        pair_id = pair_ids[key]
        ordinal = ordinals[pair_id]
        ordinals[pair_id] += 1
        records.append(
            {
                "row_id": f"r{index:07d}",
                "source_id": str(row["id"]),
                "pair_id": pair_id,
                "variant": int(row["pair_number"]),
                "bucket": int(buckets[pair_id][ordinal]),
                "concept": row["concept"],
                "antagonist": row["antagonist"],
                "concept_text": row["concept_text"],
                "antagonist_text": row["antagonist_text"],
                "genre_id": GENRES.index(row["genre"]),
                "language_id": LANGUAGES.index(row["language"]),
                "generator_id": generators.index(str(row["model"])),
                "generator": str(row["model"]),
            }
        )
    bad = {pair: count for pair, count in ordinals.items() if count != ROWS_PER_PAIR}
    if bad or len(ordinals) != len(pairs):
        raise ValueError(f"pair row counts differ from {ROWS_PER_PAIR}: {bad or 'missing pairs'}")
    semantic = records
    strata: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in sorted(semantic, key=lambda row: (row["pair_id"], row["row_id"])):
        key = (row["genre_id"], row["language_id"], row["generator_id"])
        bucket = strata.setdefault(key, [])
        if len(bucket) < VARIANTS:
            bucket.append(row)
    expected = len(GENRES) * len(LANGUAGES) * 2
    if len(strata) != expected or any(len(bucket) != VARIANTS for bucket in strata.values()):
        raise ValueError(f"syntax source grid incomplete: {len(strata)}/{expected} strata")
    sources = [row for key in sorted(strata) for row in strata[key]]
    syntax_pairs = [
        {
            "pair_id": index,
            "class_name": "Surface Form",
            "concept": positive,
            "antagonist": negative,
            "narrative_guidance": "Same source text; form only.",
        }
        for index, (_, positive, negative, _) in enumerate(TRANSFORMS)
    ]
    syntax = []
    for pair_id, (name, positive, negative, transform) in enumerate(TRANSFORMS):
        buckets = bucket_assignment(pair_id)
        for ordinal, source in enumerate(sources):
            original = source["concept_text"]
            syntax.append(
                {
                    "row_id": f"syntax:{name}:{source['row_id']}",
                    "source_id": source["row_id"],
                    "pair_id": pair_id,
                    "variant": source["variant"],
                    "bucket": int(buckets[ordinal]),
                    "concept": positive,
                    "antagonist": negative,
                    "concept_text": transform(original),
                    "antagonist_text": original,
                    "genre_id": source["genre_id"],
                    "language_id": source["language_id"],
                    "generator_id": source["generator_id"],
                    "generator": source["generator"],
                }
            )

    args.out.mkdir(parents=True, exist_ok=True)
    write_parquet(semantic, args.out / "semantic.parquet")
    write_parquet(syntax, args.out / "syntax.parquet")
    (args.out / "semantic.ontology.json").write_text(json.dumps({"pairs": pairs}, indent=2, ensure_ascii=False))
    (args.out / "syntax.ontology.json").write_text(json.dumps({"pairs": syntax_pairs}, indent=2, ensure_ascii=False))
    manifest = {
        "schema_version": 2,
        "source": args.source,
        "source_fingerprint": data._fingerprint,
        "semantic_rows": len(semantic),
        "syntax_rows": len(syntax),
        "pairs": len(pairs),
        "rows_per_pair": ROWS_PER_PAIR,
        "buckets": BUCKETS,
        "genres": list(GENRES),
        "languages": list(LANGUAGES),
        "generators": sorted({row["generator"] for row in semantic}),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    log.info("wrote %s semantic and %s syntax rows to %s", len(semantic), len(syntax), args.out)


if __name__ == "__main__":
    main()
