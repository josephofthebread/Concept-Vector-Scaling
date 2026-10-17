import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from exp_gcg.protocol import (
    MODELS,
    digest,
    load_records,
    read_json,
    save_metadata,
    write_json,
)
from exp_gcg.steering import analysis, preparation
from exp_gcg.steering.evaluation import parse_benign
from exp_gcg.steering.generation import (
    assigned,
    final_token_hook,
    jobs,
    response_path,
    resumed,
)
from exp_gcg.steering.preparation import (
    identity,
    load_benign,
    load_prepared,
    select_directions,
)


def test_selection_and_random_controls():
    base = np.array([[[1.0, 0.0], [100.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]])
    suffix = base[..., ::-1].copy()
    vectors = np.tile([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 1.0], [1.0, -1.0]], (2, 1, 1))
    pairs = [{"concept": str(i), "antagonist": str(-i)} for i in range(6)]
    selected = select_directions(base, suffix, vectors, pairs, 1, "qwen3_8b", "advbench")
    assert selected == select_directions(base, suffix, vectors, pairs, 1, "qwen3_8b", "advbench")
    assert len(selected) == 8
    first = selected[0]
    expected = vectors[1, first["feature_index"]]
    assert np.allclose(first["vector"], expected / np.linalg.norm(expected))
    base_mean, suffix_mean = base[:, 1].mean(0), suffix[:, 1].mean(0)
    expected_shift = np.dot(
        expected / np.linalg.norm(expected),
        base_mean / np.linalg.norm(base_mean) - suffix_mean / np.linalg.norm(suffix_mean),
    )
    assert first["signed_shift"] == pytest.approx(expected_shift)
    assert np.allclose(np.linalg.norm([row["vector"] for row in selected], axis=-1), 1)
    assert selected[5]["vector"] != select_directions(base, suffix, vectors, pairs, 1, "qwen3_8b", "jbb")[5]["vector"]
    with pytest.raises(ValueError):
        select_directions(base, suffix[..., :1], vectors, pairs, 1, "qwen3_8b", "advbench")


def test_hook_boundary_and_cleanup():
    blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    hidden = torch.tensor([[[3.0, 4.0], [3.0, 4.0]]])
    with final_token_hook(blocks, 2, torch.tensor([1.0, 0.0]), 0.5) as hits:
        assert torch.equal(blocks[0](hidden), hidden)
        changed = blocks[1](hidden)
        assert torch.equal(changed[:, 0], hidden[:, 0])
        assert torch.equal(changed[:, -1], torch.tensor([[5.5, 4.0]]))
        assert torch.equal(blocks[1](hidden[:, :1]), hidden[:, :1])
        assert hits == {"input": 1, "decode": 1}
    assert not blocks[1]._forward_hooks
    with pytest.raises(RuntimeError), final_token_hook(blocks, 1, torch.tensor([1.0, 0.0]), 0.5):
        raise RuntimeError("test exception cleanup")
    assert not blocks[0]._forward_hooks
    with final_token_hook(blocks, 1, torch.tensor([1.0, 0.0]), 0):
        assert torch.equal(blocks[0](hidden), hidden)


def test_parser_sharding_resume_and_input_validation(tmp_path):
    assert parse_benign('text {"category":"clarification"}') == "clarification"
    with pytest.raises(ValueError):
        parse_benign('{"category":"unknown"}')
    assert sorted(x for shard in range(3) for x in assigned(iter(range(11)), shard, 3, 0)) == list(range(11))
    assert list(assigned(iter(range(11)), 0, 3, 2)) == [0, 3]
    checkpoint = tmp_path / "checkpoint.json"
    assert not resumed(checkpoint, {"config": "a"})
    write_json(checkpoint, {"config": "a"})
    assert resumed(checkpoint, {"config": "a"})
    with pytest.raises(ValueError):
        resumed(checkpoint, {"config": "b"})
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "key": "../escape",
                "model": "qwen3_8b",
                "dataset": "advbench",
                "index": 0,
                "forbidden_prompt": "p",
                "input_ids": {},
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="safe relative"):
        load_records(records)
    benign = tmp_path / "benign.json"
    write_json(benign, [{"id": str(i), "instruction": "Question", "input": "Context"} for i in range(100)])
    assert load_benign(benign)[0]["prompt"] == "Question\n\nContext"
    write_json(benign, [{"id": "duplicate", "prompt": "Q"}] * 100)
    with pytest.raises(ValueError, match="duplicate"):
        load_benign(benign)


def test_synthetic_pipeline_and_figures(tmp_path, monkeypatch):
    from safetensors.numpy import save_file

    monkeypatch.setattr(preparation, "COUNTS", {"advbench": 1, "jbb": 1})
    monkeypatch.setattr(preparation, "load_benign", lambda _: [{"id": "alpaca/0", "prompt": "A benign question"}])
    benign = tmp_path / "benign.json"
    write_json(benign, [])
    records = []
    rng = np.random.default_rng(42)
    vectors_root, original = tmp_path / "vectors", tmp_path / "original"
    for model, (_, bank) in MODELS.items():
        root = vectors_root / bank
        root.mkdir(parents=True)
        save_file({"diff": rng.normal(size=(35, 1036, 2)).astype("float32")}, root / "diff.safetensors")
        pd.DataFrame(
            {
                "pair_id": range(1036),
                "concept": [f"c{i}" for i in range(1036)],
                "antagonist": [f"a{i}" for i in range(1036)],
            }
        ).to_parquet(root / "pairs.parquet")
        for dataset in ("advbench", "jbb"):
            key = f"{model}/{dataset}/0"
            records.append(
                {
                    "key": key,
                    "model": model,
                    "dataset": dataset,
                    "index": 0,
                    "forbidden_prompt": "Q",
                    "input_ids": {"base": [1], "suffix": [1, 2]},
                }
            )
            root = original / "records" / key
            root.mkdir(parents=True)
            provenance = {"record_sha256": identity(records[-1])}
            for condition in ("base", "suffix"):
                hidden_path = root / f"hidden_{condition}.npy"
                np.save(hidden_path, rng.normal(size=(35, 2)))
                save_metadata(hidden_path, provenance)
                response = root / "responses" / f"long_{condition}.json"
                write_json(response, {"text": "test"})
                save_metadata(response, provenance)
                score_path = root / "scores" / "finetuned" / response.name
                write_json(score_path, {"score": 0.4, "response_sha256": digest(response)})
                save_metadata(score_path, provenance)
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("\n".join(json.dumps(row) for row in records))
    output = tmp_path / "results"
    args = SimpleNamespace(records=records_path, vectors=vectors_root, original=original, benign=benign, output=output)
    preparation.prepare(args)
    preparation.prepare(args)
    prepared_path = output / "prepared.json"
    prepared = load_prepared(prepared_path)
    for model in MODELS:
        full = [identity(job) for job in jobs(prepared, model)]
        merged = [identity(job) for shard in range(3) for job in assigned(jobs(prepared, model), shard, 3, 0)]
        assert sorted(full) == sorted(merged) and len(set(merged)) == len(full)
        for job in jobs(prepared, model):
            path = response_path(output, job)
            write_json(path, {"job": job, "text": "test", "generation": {"model": model}})
            judge = "finetuned" if job["source"] == "harmful" else "gpt_oss"
            write_json(
                output / "scores" / judge / model / path.name,
                {
                    "judge": judge,
                    "score": 0.2 if judge == "finetuned" else "substantive_answer",
                    "response_sha256": digest(path),
                    "prepared_sha256": digest(prepared_path),
                    "judge_config_sha256": identity(judge),
                },
            )
    args.prepared = prepared_path
    analysis.aggregate(args)
    harmful = pd.read_csv(output / "steering_analysis" / "article_harmful.csv")
    assert set(harmful[harmful.alpha == 0]["mean"]) == {0.4}
    assert set(harmful[harmful.alpha != 0]["mean"]) == {0.2}
    benign_rows = pd.read_csv(output / "steering_analysis" / "benign_outcomes.csv")
    assert set(benign_rows[benign_rows.alpha != 0]["count"]) == {10}
    assert set(benign_rows[benign_rows.alpha == 0]["count"]) == {1}
    before = digest(output / "steering_analysis" / "sample_metrics.csv")
    analysis.aggregate(args)
    assert digest(output / "steering_analysis" / "sample_metrics.csv") == before
    args.figures_output = tmp_path / "figures"
    analysis.figures(args)
    assert (args.figures_output / "gcg-steering-advbench.pdf").is_file()
    assert (args.figures_output / "gcg-steering-advbench.png").is_file()
    missing = next((output / "scores").rglob("*.json"))
    score = read_json(missing)
    missing.unlink()
    with pytest.raises(FileNotFoundError):
        analysis.aggregate(args)
    write_json(missing, score)
    response = next((output / "responses").rglob("*.json"))
    saved = read_json(response)
    saved["text"] = "modified"
    write_json(response, saved)
    with pytest.raises(ValueError, match="stale"):
        analysis.aggregate(args)
    assert not read_json(output / "steering_analysis" / "manifest.json")["complete"]
    prepared["random_seed"] += 1
    write_json(prepared_path, prepared)
    with pytest.raises(ValueError, match="modified"):
        load_prepared(prepared_path)


def test_bounded_generation_and_resume(tmp_path, monkeypatch):
    from exp_gcg.steering import generation

    class TinyTokenizer:
        pad_token_id = 0

        def decode(self, ids, **kwargs):
            return " ".join(map(str, ids))

    class TinyModel:
        device = torch.device("cpu")
        config = SimpleNamespace(max_position_embeddings=1024)
        model = SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Identity()]))
        calls = 0

        def generate(self, input_ids, **kwargs):
            self.calls += 1
            hidden = torch.tensor([[[3.0, 4.0], [3.0, 4.0]]])
            first = self.model.layers[0](hidden)
            self.model.layers[0](hidden[:, :1])
            token = first[:, -1].argmax(-1).reshape(1, 1)
            return torch.cat([input_ids, token], dim=1)

    model = TinyModel()
    inputs = torch.tensor([[1, 2]])
    plain = model.generate(inputs)
    with final_token_hook(model.model.layers, 1, torch.tensor([1.0, 0.0]), 0):
        assert torch.equal(model.generate(inputs), plain)
    prepared = {
        "records": [
            {
                "model": "qwen3_8b",
                "dataset": "advbench",
                "key": "record",
                "forbidden_prompt": "Q",
                "input_ids": {"base": [1, 2], "suffix": [1, 3]},
            }
        ],
        "directions": [
            {
                "model": "qwen3_8b",
                "dataset": "advbench",
                "kind": "concept",
                "layer": 1,
                "label": "test",
                "vector": [1.0, 0.0],
            }
        ],
        "alphas": [0.5],
        "benign": [],
    }
    monkeypatch.setattr(generation, "load_prepared", lambda _: prepared)
    monkeypatch.setattr(generation, "load_model", lambda *args, **kwargs: (TinyTokenizer(), model))
    source = tmp_path / "prepared.json"
    write_json(source, prepared)
    args = SimpleNamespace(prepared=source, output=tmp_path, model="qwen3_8b", device="cpu", shard=0, shards=1, limit=1)
    generation.generate(args)
    calls = model.calls
    generation.generate(args)
    assert model.calls == calls
    response = read_json(next((tmp_path / "responses").rglob("*.json")))
    assert response["hook_calls"] == {"input": 1, "decode": 1}
    assert response["generation_token_ids"] == [0]
    args.qwen3_8b = "different-snapshot"
    with pytest.raises(ValueError, match="stale"):
        generation.generate(args)


def test_bounded_benign_judge_and_resume(tmp_path, monkeypatch):
    from exp_gcg.steering import evaluation

    class TinyTokenizer:
        pad_token_id = 0

        def apply_chat_template(self, *args, **kwargs):
            return "<|start|>assistant"

        def __call__(self, *args, **kwargs):
            return {"input_ids": [1, 2]}

        def decode(self, *args, **kwargs):
            return '{"category":"substantive_answer"}'

    class TinyJudge:
        device = torch.device("cpu")
        config = SimpleNamespace(max_position_embeddings=1024)
        calls = 0

        def generate(self, input_ids, **kwargs):
            self.calls += 1
            return torch.cat([input_ids, torch.tensor([[3]])], dim=1)

    model = TinyJudge()
    prepared = {"records": [], "directions": [], "benign": [{"id": "alpaca/0", "prompt": "Q"}]}
    monkeypatch.setattr(evaluation, "load_prepared", lambda _: prepared)
    monkeypatch.setattr(evaluation, "load_model", lambda *args, **kwargs: (TinyTokenizer(), model))
    path = tmp_path / "prepared.json"
    write_json(path, prepared)
    job = next(jobs(prepared, "qwen3_8b"))
    write_json(response_path(tmp_path, job), {"job": job, "text": "A"})
    args = SimpleNamespace(
        prepared=path,
        output=tmp_path,
        model="qwen3_8b",
        device="cpu",
        shard=0,
        shards=1,
        limit=1,
        judge="gpt_oss",
        gpt_oss="tiny-judge",
    )
    evaluation.judge(args)
    assert model.calls == 1
    evaluation.judge(args)
    assert model.calls == 1
    score = read_json(next((tmp_path / "scores").rglob("*.json")))
    assert score["score"] == "substantive_answer" and score["schema_version"] == 1
    args.gpt_oss = "changed-judge"
    with pytest.raises(ValueError, match="stale"):
        evaluation.judge(args)
