from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .ontology import iter_feature_pairs, load_ontology


def slug_part(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", (value or "").strip().lower())
    return text.strip("_") or "x"


def make_row_id(
    model: str,
    class_name: str,
    concept: str,
    antagonist: str,
    genre: str,
    language: str,
    pair_number: int,
) -> str:
    return "__".join(
        [
            slug_part(model.replace("/", "_")),
            slug_part(class_name)[:48],
            slug_part(concept),
            slug_part(antagonist),
            slug_part(genre),
            slug_part(language),
            str(int(pair_number)),
        ]
    )


def guidance_maps(ontology_path: Path) -> tuple[dict[tuple[int, int], str], dict[tuple[str, str], str]]:
    by_idx: dict[tuple[int, int], str] = {}
    by_labels: dict[tuple[str, str], str] = {}
    for pair in iter_feature_pairs(load_ontology(ontology_path)):
        by_idx[(pair.class_id, pair.pair_index)] = pair.narrative_guidance
        by_labels[(pair.concept, pair.antagonist)] = pair.narrative_guidance
    return by_idx, by_labels


def resolve_guidance(
    row: dict[str, Any],
    by_idx: dict[tuple[int, int], str],
    by_labels: dict[tuple[str, str], str],
) -> str:
    try:
        key = (int(row["class_id"]), int(row["pair_index"]))
        if key in by_idx:
            return by_idx[key]
    except (KeyError, TypeError, ValueError):
        pass
    return by_labels.get((str(row.get("concept", "")), str(row.get("antagonist", ""))), "")


def pair_story_rows(
    rows: list[dict[str, Any]],
    model: str,
    by_idx: dict[tuple[int, int], str],
    by_labels: dict[tuple[str, str], str],
) -> list[dict[str, Any]]:
    pending: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    paired: list[dict[str, Any]] = []
    for row in rows:
        pole = str(row.get("pole") or "")
        if pole not in {"concept", "antagonist"}:
            continue
        try:
            pair_number = int(row.get("pair_number", 0))
        except (TypeError, ValueError):
            pair_number = 0
        key = (str(row.get("request_id") or ""), pair_number)
        pending[key][pole] = row
        poles = pending[key]
        if "concept" not in poles or "antagonist" not in poles:
            continue
        concept_row = poles["concept"]
        class_name = str(concept_row.get("class_name") or "")
        concept = str(concept_row.get("concept") or "")
        antagonist = str(concept_row.get("antagonist") or "")
        genre = str(concept_row.get("genre") or "")
        language = str(concept_row.get("language") or "")
        paired.append(
            {
                "id": make_row_id(model, class_name, concept, antagonist, genre, language, pair_number),
                "pair_number": pair_number,
                "concept": concept,
                "antagonist": antagonist,
                "concept_text": str(concept_row.get("text") or ""),
                "antagonist_text": str(poles["antagonist"].get("text") or ""),
                "narrative_guidance": resolve_guidance(concept_row, by_idx, by_labels),
                "class_name": class_name,
                "language": language,
                "genre": genre,
                "model": model,
                "shared_setup": str(concept_row.get("shared_setup") or ""),
            }
        )
        del pending[key]
    return paired


def pair_stories_jsonl(stories_path: Path, out_path: Path, model: str, ontology_path: Path) -> int:
    by_idx, by_labels = guidance_maps(ontology_path)
    rows = []
    with stories_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    paired = pair_story_rows(rows, model, by_idx, by_labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in paired:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(paired)
