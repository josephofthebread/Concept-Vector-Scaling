import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from exp_gcg import analysis, evaluation, generation
from exp_gcg.protocol import (
    check_checkpoint,
    identity,
    read_json,
    save_metadata,
    write_json,
)


class TinyTokenizer:
    pad_token_id = 0
    chat_template = "synthetic"

    def get_vocab(self):
        return {str(i): i - 1 for i in range(1, 6)}

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text)))}

    def decode(self, tokens, **kwargs):
        return "x" * len(tokens)

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        return [1, 2] if tokenize else "<|start|>assistant"


class TinyConfig:
    max_position_embeddings = 1024
    _commit_hash = "synthetic"

    def to_dict(self):
        return {"max_position_embeddings": self.max_position_embeddings}


class TinyGenerationConfig:
    eos_token_id = 0

    def to_dict(self):
        return {"eos_token_id": self.eos_token_id}


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        first = (input_ids == 1).float()
        hidden = torch.stack([first, 1 - first], dim=-1)
        return SimpleNamespace(hidden_states=(hidden, hidden), last_hidden_state=hidden)


class TinyModel:
    device = torch.device("cpu")
    config = TinyConfig()
    generation_config = TinyGenerationConfig()

    def __init__(self):
        self.model = TinyBackbone()
        self.requests = []

    def generate(self, input_ids, max_new_tokens, stopping_criteria=None, **kwargs):
        self.requests.append({"maximum": max_new_tokens, "stopping": bool(stopping_criteria)})
        if not stopping_criteria:
            return torch.cat([input_ids, torch.full((1, max_new_tokens), 3)], dim=1)
        sequence = input_ids
        for _ in range(max_new_tokens):
            sequence = torch.cat([sequence, torch.tensor([[3]])], dim=1)
            if stopping_criteria(sequence, None):
                break
        return sequence


@pytest.fixture
def suffix_run(tmp_path, monkeypatch):
    tokenizer, model = TinyTokenizer(), TinyModel()
    record = {
        "key": "record",
        "model": "qwen3_8b",
        "dataset": "advbench",
        "index": 0,
        "forbidden_prompt": "Q",
        "input_ids": {"base": [1], "suffix": [2]},
    }
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps(record) + "\n")
    template = tmp_path / "templates.json"
    write_json(template, {"strongreject_finetuned": "{forbidden_prompt}{response}"})
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "weights.safetensors").write_bytes(b"synthetic model")
    args = SimpleNamespace(
        records=records,
        model="qwen3_8b",
        qwen3_8b=str(snapshot),
        device="cpu",
        strongreject="synthetic-adapter",
        templates=template,
        output=tmp_path / "output",
        max_new_tokens=None,
        strongreject_base="synthetic-base",
    )
    monkeypatch.setattr(generation, "load_model", lambda *args, **kwargs: (tokenizer, model))
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(generation, "CONTEXT", 16)
    return args, record, tokenizer, model


def test_reference_stopping_and_resumption(suffix_run):
    args, record, _tokenizer, model = suffix_run
    generation.generate(args)
    root = args.output / "records" / record["key"]
    response = read_json(root / "responses" / "long_base.json")
    assert response["max_new_tokens"] == 1023
    assert response["stop_reason"] == "judge_context"
    assert len(response["generation_token_ids"]) == 14
    calls, hidden_calls = len(model.requests), model.model.calls
    generation.generate(args)
    assert len(model.requests) == calls and model.model.calls == hidden_calls
    args.max_new_tokens = 5
    with pytest.raises(ValueError, match="stale"):
        generation.generate(args)


@pytest.mark.parametrize("change", ["record", "weights", "response", "hidden", "template", "legacy"])
def test_generation_rejects_changed_checkpoints(suffix_run, change):
    args, record, _tokenizer, _model = suffix_run
    generation.generate(args)
    root = args.output / "records" / record["key"]
    if change == "record":
        record["input_ids"]["base"] = [4]
        args.records.write_text(json.dumps(record) + "\n")
    elif change == "weights":
        from pathlib import Path

        (Path(args.qwen3_8b) / "weights.safetensors").write_bytes(b"different weights")
    elif change == "response":
        path = root / "responses" / "long_base.json"
        saved = read_json(path)
        saved["text"] = "changed"
        write_json(path, saved)
    elif change == "hidden":
        np.save(root / "hidden_base.npy", np.zeros((2, 2)))
    elif change == "template":
        write_json(args.templates, {"strongreject_finetuned": "new {forbidden_prompt}{response}"})
    else:
        (root / "hidden_base.npy.meta.json").unlink()
    with pytest.raises(ValueError, match="stale|legacy"):
        generation.generate(args)


def test_judging_arbitrary_keys_and_aggregation(suffix_run, monkeypatch):
    from safetensors.numpy import save_file

    args, _record, tokenizer, model = suffix_run
    generation.generate(args)

    class TinyAdapter:
        device = torch.device("cpu")

        def __init__(self, base):
            head = torch.nn.Linear(2, 5, bias=False)
            with torch.no_grad():
                head.weight.copy_(torch.arange(10).reshape(5, 2))
            self.base_model = SimpleNamespace(model=SimpleNamespace(model=base.model, lm_head=head))

        def eval(self):
            return self

    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=lambda base, *args, **kwargs: TinyAdapter(base))),
    )
    monkeypatch.setattr(evaluation, "load_model", lambda *args, **kwargs: (tokenizer, model))
    evaluation.judge(args)
    score_path = args.output / "records" / "record" / "scores" / "finetuned" / "long_base.json"
    assert score_path.exists()
    assert 0 <= read_json(score_path)["score"] <= 1
    calls = model.model.calls
    evaluation.judge(args)
    assert model.model.calls == calls
    original_adapter = args.strongreject
    args.strongreject = "changed-adapter"
    with pytest.raises(ValueError, match="stale"):
        evaluation.judge(args)
    args.strongreject = original_adapter

    args.vectors = args.output / "banks"
    bank = args.vectors / "bank"
    bank.mkdir(parents=True)
    vectors = np.tile(np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]]), (2, 1, 1))
    save_file({"diff": vectors}, bank / "diff.safetensors")
    pd.DataFrame({"pair_id": range(5), "concept": list("abcde"), "antagonist": list("ABCDE")}).to_parquet(
        bank / "pairs.parquet"
    )
    monkeypatch.setattr(analysis, "MODELS", {"qwen3_8b": ("synthetic", "bank")})
    monkeypatch.setattr(analysis, "COUNTS", {"advbench": 1})
    analysis.aggregate(args)
    rankings = pd.read_csv(args.output / "analysis" / "rankings.csv")
    first = rankings.iloc[0]
    assert first.pair_id == 0 and first.signed_change == pytest.approx(1.0)
    assert first.absolute_change == pytest.approx(1.0)
    validation = read_json(args.output / "analysis" / "validation.json")
    assert validation["schema_version"] == 2 and validation["complete"]

    score = read_json(score_path)
    score["score"] = 0.1
    write_json(score_path, score)
    with pytest.raises(ValueError, match="stale"):
        evaluation.judge(args)
    with pytest.raises(ValueError, match="stale"):
        analysis.aggregate(args)
    assert not read_json(args.output / "analysis" / "validation.json")["complete"]


def test_metadata_binds_records_and_contents(tmp_path):
    path = tmp_path / "checkpoint.json"
    assert check_checkpoint(path) is None
    write_json(path, {"text": "test"})
    with pytest.raises(ValueError, match="legacy"):
        check_checkpoint(path)
    save_metadata(path, {"record_sha256": identity({"record": 1})})
    assert check_checkpoint(path, {"record_sha256": identity({"record": 1})})
    with pytest.raises(ValueError, match="stale"):
        check_checkpoint(path, {"record_sha256": identity({"record": 2})})
