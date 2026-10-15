from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import stable_id


@dataclass(frozen=True)
class FeaturePair:
    class_id: int
    class_name: str
    pair_index: int
    concept: str
    antagonist: str
    narrative_guidance: str

    @property
    def pair_id(self) -> str:
        return stable_id(self.class_id, self.pair_index, self.concept, self.antagonist)


def load_ontology(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if "classes" not in obj or not isinstance(obj["classes"], list):
        raise ValueError(f"Ontology has no classes list: {path}")
    return obj


def iter_feature_pairs(ontology: dict[str, Any], class_ids: set[int] | None = None) -> list[FeaturePair]:
    pairs: list[FeaturePair] = []
    for cls in ontology["classes"]:
        class_id = int(cls["id"])
        if class_ids is not None and class_id not in class_ids:
            continue
        class_name = str(cls["name"])
        for idx, pair in enumerate(cls.get("pairs", [])):
            pairs.append(
                FeaturePair(
                    class_id=class_id,
                    class_name=class_name,
                    pair_index=idx,
                    concept=str(pair["concept"]),
                    antagonist=str(pair["antagonist"]),
                    narrative_guidance=str(pair.get("narrative_guidance", "")),
                )
            )
    return pairs


def select_pairs(
    ontology_path: str | Path,
    class_ids: list[int],
    pair_index: int | None = None,
) -> list[FeaturePair]:
    ontology = load_ontology(ontology_path)
    pairs = iter_feature_pairs(ontology, set(class_ids))
    if pair_index is not None:
        pairs = [p for p in pairs if p.pair_index == pair_index]
    if not pairs:
        raise ValueError(f"No feature pairs matched class_ids={class_ids}, pair_index={pair_index}")
    return pairs


def class_names(ontology_path: str | Path, class_ids: list[int]) -> dict[int, str]:
    ontology = load_ontology(ontology_path)
    wanted = set(class_ids)
    return {int(cls["id"]): str(cls["name"]) for cls in ontology["classes"] if int(cls["id"]) in wanted}
