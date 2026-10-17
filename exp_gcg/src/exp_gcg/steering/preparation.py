import json

import numpy as np
import pandas as pd

from ..protocol import (
    COUNTS,
    MODELS,
    check_checkpoint,
    digest,
    identity,
    load_records,
    normalize,
    read_json,
    write_json,
)

LAYERS = {"qwen3_8b": 29, "llama3_1_8b_instruct": 26, "gemma2_9b_it": 34}
ALPHAS = (-0.5, -0.25, -0.1, 0.1, 0.25, 0.5)
CATEGORIES = ("safety_refusal", "capability_limit", "clarification", "substantive_answer")


def checked_score(path, response):
    saved = read_json(path)
    value = saved["score"]
    if saved["response_sha256"] != digest(response):
        raise ValueError(f"stale response score: {path.name}")
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("score must be finite and in [0,1]")
    return float(value)


def select_directions(base, suffix, vectors, pairs, layer, model, dataset):
    """Rank mean-before-cosine shifts at the fixed layer; retain original orientation."""
    if (
        base.ndim != 3
        or base.shape != suffix.shape
        or vectors.ndim != 3
        or base.shape[1:] != (vectors.shape[0], vectors.shape[2])
        or len(pairs) != vectors.shape[1]
        or len(pairs) < 5
        or not 0 < layer < vectors.shape[0]
    ):
        raise ValueError("incompatible residual/vector/feature dimensions")
    if not np.isfinite(base).all() or not np.isfinite(suffix).all():
        raise ValueError("non-finite residuals")
    directions = normalize(vectors[layer].astype("float64"))
    change = directions @ (normalize(base[:, layer].mean(0)) - normalize(suffix[:, layer].mean(0)))
    order = np.lexsort((np.arange(len(pairs)), -np.abs(change)))[:5]
    result = []
    for rank, index in enumerate(order, 1):
        pair = pairs[index]
        result.append(
            {
                "model": model,
                "dataset": dataset,
                "kind": "concept",
                "layer": layer,
                "feature_index": int(index),
                "pair_id": pair.get("pair_id", int(index)),
                "label": pair["concept"],
                "antagonist": pair["antagonist"],
                "rank": rank,
                "signed_shift": float(change[index]),
                "vector": directions[index].tolist(),
            }
        )
    rng = np.random.default_rng(20260914 + list(MODELS).index(model) * 10 + list(COUNTS).index(dataset))
    for draw in range(3):
        result.append(
            {
                "model": model,
                "dataset": dataset,
                "kind": "random_direction",
                "layer": layer,
                "label": f"random_{draw}",
                "draw": draw,
                "vector": normalize(rng.normal(size=vectors.shape[-1]).astype("float32")).tolist(),
            }
        )
    return result


def load_benign(path):
    rows = read_json(path) if path.suffix == ".json" else [json.loads(line) for line in path.read_text().splitlines()]
    if not isinstance(rows, list) or len(rows) != 100:
        raise ValueError("supply the original 100 Alpaca records")
    result = []
    for row in rows:
        prompt = row.get("prompt")
        if prompt is None:
            prompt = row["instruction"] + ("\n\n" + row["input"] if row.get("input") else "")
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(row.get("id"), str):
            raise ValueError("benign rows require a unique string id and nonempty prompt")
        result.append({"id": row["id"], "prompt": prompt})
    if len({row["id"] for row in result}) != 100:
        raise ValueError("duplicate benign record ids")
    return result


def prepare(args):
    from safetensors.numpy import load_file

    records = load_records(args.records)
    benign = load_benign(args.benign)
    hashes = {"records": digest(args.records), "benign": digest(args.benign)}
    directions, baselines = [], []
    for model, (_, bank) in MODELS.items():
        bank_root = args.vectors / bank
        pairs_path, vectors_path = bank_root / "pairs.parquet", bank_root / "diff.safetensors"
        pairs = pd.read_parquet(pairs_path).to_dict("records")
        vectors = load_file(vectors_path)["diff"]
        if vectors.ndim != 3 or vectors.shape[1] != 1036:
            raise ValueError("article feature banks require 1,036 directions")
        if len({pair["pair_id"] for pair in pairs}) != len(pairs):
            raise ValueError("duplicate pair ids")
        hashes[f"{model}/pairs"] = digest(pairs_path)
        hashes[f"{model}/vectors"] = digest(vectors_path)
        for dataset, expected in COUNTS.items():
            group = sorted(
                (r for r in records if r["model"] == model and r["dataset"] == dataset), key=lambda r: r["index"]
            )
            if len(group) != expected or [r["index"] for r in group] != list(range(expected)):
                raise ValueError(f"incomplete article group: {model}/{dataset}")
            states = {condition: [] for condition in ("base", "suffix")}
            for record in group:
                root = args.original / "records" / record["key"]
                expected_record = {"record_sha256": identity(record)}
                for condition, hidden_rows in states.items():
                    hidden_path = root / f"hidden_{condition}.npy"
                    if check_checkpoint(hidden_path, expected_record) is None:
                        raise FileNotFoundError(hidden_path)
                    hidden_rows.append(np.load(hidden_path, allow_pickle=False))
                    hashes[f"{record['key']}/hidden_{condition}"] = digest(hidden_path)
                    response = root / "responses" / f"long_{condition}.json"
                    score = root / "scores" / "finetuned" / response.name
                    for path in (response, score):
                        if check_checkpoint(path, expected_record) is None:
                            raise FileNotFoundError(path)
                    value = checked_score(score, response)
                    hashes[f"{record['key']}/response_{condition}"] = digest(response)
                    hashes[f"{record['key']}/score_{condition}"] = digest(score)
                    baselines.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "record": record["key"],
                            "condition": condition,
                            "score": value,
                        }
                    )
            directions.extend(
                select_directions(
                    np.stack(states["base"]).astype("float64"),
                    np.stack(states["suffix"]).astype("float64"),
                    vectors,
                    pairs,
                    LAYERS[model],
                    model,
                    dataset,
                )
            )
    value = {
        "schema_version": 1,
        "complete": True,
        "alphas": ALPHAS,
        "max_new_tokens": 512,
        "shift_convention": "base_minus_suffix",
        "layers": LAYERS,
        "random_seed": 20260914,
        "records": records,
        "benign": benign,
        "directions": directions,
        "baselines": baselines,
        "input_hashes": hashes,
    }
    path = args.output / "prepared.json"
    if path.exists():
        if read_json(path) != json.loads(json.dumps(value)):
            raise ValueError("prepared inputs changed; use a new output directory")
    else:
        write_json(path, value)
    write_json(
        args.output / "preparation_manifest.json",
        {
            "schema_version": 1,
            "complete": True,
            "prepared_sha256": digest(path),
            "configuration_sha256": identity(value),
        },
    )


def load_prepared(path):
    value = read_json(path)
    manifest = read_json(path.parent / "preparation_manifest.json")
    if (
        value.get("schema_version") != 1
        or not value.get("complete")
        or value.get("shift_convention") != "base_minus_suffix"
        or manifest.get("prepared_sha256") != digest(path)
        or manifest.get("configuration_sha256") != identity(value)
    ):
        raise ValueError("invalid or modified prepared artifact")
    return value
