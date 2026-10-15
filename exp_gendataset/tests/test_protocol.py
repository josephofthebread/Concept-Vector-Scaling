from __future__ import annotations

import json
from pathlib import Path

from exp_gendataset.cli import load_config, resolve
from exp_gendataset.generation import (
    MAX_VALIDATION_RETRIES,
    build_generation_tasks,
    build_story_prompt,
    build_story_repair_prompt,
    genre_length_hint,
)
from exp_gendataset.ontology import iter_feature_pairs, load_ontology
from exp_gendataset.pairing import guidance_maps, pair_story_rows
from exp_gendataset.story_few_shot import few_shot_section

ROOT = Path(__file__).resolve().parents[1]


def test_ontology_matches_paper():
    ontology = load_ontology(ROOT / "data" / "ontology.json")
    pairs = iter_feature_pairs(ontology)
    classes = {p.class_id for p in pairs}
    assert len(classes) == 148
    assert len(pairs) == 1036
    assert all(p.narrative_guidance.strip() for p in pairs)
    example = next(p for p in pairs if p.concept == "genuine goal pursuit" and p.antagonist == "reward hacking")
    assert example.narrative_guidance.startswith("One agent works earnestly toward the intended objective")


def test_matrix_counts():
    cfg = load_config(resolve("data/paper.yaml"))
    pairs = iter_feature_pairs(load_ontology(resolve(cfg["ontology_path"])))
    tasks = build_generation_tasks(
        pairs,
        cfg["genres"],
        cfg["languages"],
        stories_per_pole=cfg["generation"]["stories_per_pole"],
        target_words_min=cfg["generation"]["target_words_min"],
        target_words_max=cfg["generation"]["target_words_max"],
    )
    assert cfg["generation"]["temperature"] == 1.0
    assert cfg["generation"]["top_p"] == 0.95
    assert cfg["generation"]["stories_per_pole"] == 10
    assert cfg["generation"]["validation_retries"] == MAX_VALIDATION_RETRIES == 10
    assert cfg["models"] == [
        "qwen/qwen3-235b-a22b-2507",
        "deepseek/deepseek-v4-flash",
    ]
    assert len(tasks) == 1036 * 10 * 5
    assert len(tasks) * 10 * 2 * 2 == 2_072_000


def test_prompt_matches_appendix_b():
    cfg = load_config(resolve("data/paper.yaml"))
    pairs = iter_feature_pairs(load_ontology(resolve(cfg["ontology_path"])))
    pair = next(p for p in pairs if p.concept == "genuine goal pursuit")
    task = build_generation_tasks(
        [pair],
        ["speech"],
        ["Russian"],
        stories_per_pole=10,
        target_words_min=200,
        target_words_max=400,
    )[0]
    messages = build_story_prompt(task)
    assert [m["role"] for m in messages] == ["system", "user"]
    user = messages[1]["content"]
    assert "genuine goal pursuit" in user
    assert "reward hacking" in user
    assert pair.class_name in user
    assert pair.narrative_guidance in user
    assert "speech" in user and "Russian" in user
    assert "200-400 words (target 300-380)" in user
    assert "universal access vs privileged access" in user
    assert "cause search vs blame assignment" in user
    assert "Коллеги, благодарю, что собрались сегодня" in user
    assert "4月8日" in user
    assert few_shot_section() in user
    assert "pair_number 0..9" in user
    repair = build_story_repair_prompt(task, {"story_pairs": []}, [3, 7])
    assert "3, 7" in repair[1]["content"]
    assert genre_length_hint("diary", "Mandarin Chinese") == (
        "Each story: 600-1200 characters (target 900-1100); below 600 is invalid"
    )
    assert genre_length_hint("diary", "English").startswith("Each story: 200-400 words")


def test_pairing_schema():
    ontology = ROOT / "data" / "ontology.json"
    by_idx, by_labels = guidance_maps(ontology)
    rows = [
        {
            "request_id": "req",
            "class_id": 2,
            "pair_index": 0,
            "pair_number": 0,
            "pole": "concept",
            "class_name": "Alignment & Misalignment Behaviors",
            "concept": "genuine goal pursuit",
            "antagonist": "reward hacking",
            "genre": "narrative_3rd_person",
            "language": "English",
            "text": "concept story",
            "shared_setup": "A shared English setup.",
        },
        {
            "request_id": "req",
            "pair_number": 0,
            "pole": "antagonist",
            "text": "antagonist story",
        },
    ]
    paired = pair_story_rows(rows, "qwen/qwen3-235b-a22b-2507", by_idx, by_labels)
    assert len(paired) == 1
    row = paired[0]
    assert set(row) == {
        "id",
        "pair_number",
        "concept",
        "antagonist",
        "concept_text",
        "antagonist_text",
        "narrative_guidance",
        "class_name",
        "language",
        "genre",
        "model",
        "shared_setup",
    }
    assert row["concept_text"] == "concept story"
    assert row["antagonist_text"] == "antagonist story"
    assert row["narrative_guidance"].startswith("One agent works earnestly")
    assert json.dumps(row)
