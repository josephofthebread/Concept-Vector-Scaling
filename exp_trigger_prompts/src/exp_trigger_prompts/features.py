from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import KIND_COUNTS, SCENARIO_HINTS, SEEDS

DATA = Path(__file__).resolve().parents[2] / "data" / "features.json"


@dataclass(frozen=True)
class Feature:
    pair_id: int
    class_name: str
    concept: str
    antagonist: str
    narrative_guidance: str


@dataclass(frozen=True)
class Item:
    pair_id: int
    class_name: str
    kind: str
    concept: str
    antagonist: str
    guidance: str
    slot: int
    seed: int
    hint: str

    @property
    def pole_text(self) -> str:
        """The pole an embodiment draft must express; empty for neutral drafts."""
        if self.kind == "concept":
            return self.concept
        if self.kind == "antagonist":
            return self.antagonist
        return ""

    @property
    def opposite_text(self) -> str:
        if self.kind == "concept":
            return self.antagonist
        if self.kind == "antagonist":
            return self.concept
        return ""


def load_features(path: Path = DATA) -> list[Feature]:
    payload = json.loads(path.read_text())
    return [Feature(**record) for record in payload["features"]]


def build_items(features: list[Feature]) -> list[Item]:
    """Expand each feature into 64 draft items in a fixed (feature, kind, slot) order."""
    items: list[Item] = []
    for feature in features:
        for kind, count in KIND_COUNTS.items():
            for slot in range(count):
                items.append(
                    Item(
                        pair_id=feature.pair_id,
                        class_name=feature.class_name,
                        kind=kind,
                        concept=feature.concept,
                        antagonist=feature.antagonist,
                        guidance=feature.narrative_guidance,
                        slot=slot,
                        seed=SEEDS[slot],
                        hint=SCENARIO_HINTS[slot],
                    )
                )
    return items
