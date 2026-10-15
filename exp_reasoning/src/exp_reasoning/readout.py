from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp_reasoning.benchmarks import MODEL_ID, instruction, load_benchmark, thinking_span
from exp_reasoning.generate import GENERATION_VERSION, benchmark_names
from exp_reasoning.stats import pair_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Directory holding generations*.jsonl")
    parser.add_argument("--vector-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Where to write the scores (default: --results)")
    parser.add_argument(
        "--benchmark",
        default="gpqa_diamond",
        help="One benchmark, a comma-separated list to pool, or 'all'",
    )
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=None, help="Skip traces longer than this")
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=1024,
        help="How many tokens go through the model at once. Lower it if the card still runs out.",
    )
    parser.add_argument(
        "--include-unfinished",
        action="store_true",
        help="Also replay unclosed thinking blocks and count them in the incorrect group.",
    )
    return parser.parse_args()


def selected_benchmark_names(value: str) -> tuple[str, ...]:
    selected = benchmark_selection(value)
    return benchmark_names("all") if selected is None else selected


def load_records(
    results: Path,
    benchmarks: tuple[str, ...],
    include_unfinished: bool = False,
) -> list[dict[str, Any]]:
    records = load(results, ",".join(benchmarks), GENERATION_VERSION)
    rows = [row for row in records if include_unfinished or row["reasoning_status"] == "closed_thinking"]
    if not rows:
        raise SystemExit(f"No usable records for {','.join(benchmarks)}")
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: item["key"]):
        unique.setdefault((row["benchmark"], row["id"]), row)
    return list(unique.values())


def load_directions(vector_dir: Path, layer: int, pairs: list[int], device: torch.device) -> torch.Tensor:
    manifest = vector_manifest(vector_dir)
    try:
        position = vector_layer_index(manifest, layer)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    tensor = load_file(vector_dir / "diff.safetensors")["diff"]
    stacked = torch.stack([tensor[position, pair].float() for pair in pairs])
    return F.normalize(stacked, dim=-1).to(device=device, dtype=torch.float16)


def benchmark_selection(value: str | None) -> tuple[str, ...] | None:
    """Turn the CLI spelling into a benchmark filter; None means pool everything present."""
    if value is None or value.strip().lower() == "all":
        return None
    selected = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not selected:
        raise ValueError("--benchmark must name at least one benchmark")
    return selected


def load(results: Path, benchmark: str | None, version: int | None) -> list[dict[str, Any]]:
    everything = []
    for path in sorted(results.glob("generations*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    everything.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"{path.name}:{number}: skipping an unreadable line")
    versions = collections.Counter(record.get("generation_version") for record in everything)
    known = [value for value in versions if value is not None]
    if not known:
        raise SystemExit(f"No versioned records found under {results}")
    wanted = version if version is not None else max(known)
    records = {record["key"]: dict(record) for record in everything if record.get("generation_version") == wanted}
    selected = benchmark_selection(benchmark)
    rows = [row for row in records.values() if selected is None or row["benchmark"] in selected]
    if not rows:
        raise SystemExit(f"No generation_version {wanted} records found under {results}")
    missing_scores = sum(row.get("correct") is None for row in rows)
    if missing_scores:
        raise SystemExit(
            f"{missing_scores} selected records have correct=null (math-verify missing at generation time)"
        )
    skipped = sum(count for value, count in versions.items() if value != wanted)
    print(f"generation_version {wanted}: {len(rows)} records (skipped {skipped} from other versions)")
    return rows


def vector_manifest(vector_dir: Path) -> dict[str, Any]:
    manifest = json.loads((vector_dir / "manifest.json").read_text(encoding="utf-8"))
    pairs = manifest.get("pairs", manifest.get("n_pairs"))
    layers = manifest.get("layers")
    valid_layers = isinstance(layers, int) or (
        isinstance(layers, list) and all(isinstance(layer, int) for layer in layers)
    )
    if manifest.get("model") != MODEL_ID or pairs != 1036 or manifest.get("hidden_size") != 4096 or not valid_layers:
        raise ValueError(f"Unexpected vector manifest at {vector_dir}: {manifest}")
    if not manifest.get("capture_merge_key"):
        raise ValueError(f"Vector manifest at {vector_dir} has no capture_merge_key")
    return manifest


def vector_layer_index(manifest: dict[str, Any], layer: int) -> int:
    """Map a decoder-block number to the tensor axis in old or compact artifacts."""
    layers = manifest["layers"]
    if isinstance(layers, int):
        if not 0 <= layer < layers:
            raise ValueError(f"Layer {layer} is not present (available: 0..{layers - 1})")
        return layer
    if layer not in layers:
        raise ValueError(f"Layer {layer} is not present (available: {layers})")
    return layers.index(layer)


class Capture:
    """Collects the residual stream of one block for a whole teacher-forced sequence."""

    def __init__(self, model: Any, layer: int) -> None:
        self.value: torch.Tensor | None = None
        self.handle = model.model.layers[layer - 1].register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        residual = output[0] if isinstance(output, tuple) else output
        self.value = residual[0].detach()

    def close(self) -> None:
        self.handle.remove()


@torch.inference_mode()
def trace_cosines(
    model: Any,
    tokenizer: Any,
    capture: Capture,
    record: dict[str, Any],
    prompt: str,
    directions: torch.Tensor,
    chunk_tokens: int,
) -> np.ndarray | None:
    """Per-reasoning-token cosine against each direction, shaped (pairs, reasoning tokens)."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    prompt_ids = tokenizer([rendered], return_tensors="pt").input_ids
    continuation_ids = tokenizer(record["output"], add_special_tokens=False, return_tensors="pt").input_ids
    token_ids = continuation_ids[0].tolist()
    start, end, status = thinking_span(tokenizer, token_ids, len(token_ids))
    if status not in {"closed_thinking", "unclosed_thinking"} or start is None or end <= start:
        return None
    sequence = torch.cat([prompt_ids, continuation_ids[:, :end]], dim=1).to(model.device)
    offset = prompt_ids.shape[1]

    captured, cache = [], None
    for position in range(0, sequence.shape[1], chunk_tokens):
        output = model.model(sequence[:, position : position + chunk_tokens], past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        if capture.value is None:
            return None
        captured.append(capture.value)
    hidden = torch.cat(captured, dim=0)[offset + start : offset + end]
    hidden = F.normalize(hidden.float(), dim=-1).to(dtype=directions.dtype)
    return (hidden @ directions.T).float().cpu().numpy().T


def binned_all(values: np.ndarray, bins: int) -> np.ndarray:
    """Average every concept into equal relative-position bins, shaped (pairs, bins)."""
    edges = np.linspace(0, values.shape[1], bins + 1).astype(int)
    starts = np.minimum(edges[:-1], values.shape[1] - 1)
    counts = np.maximum(np.diff(edges), 1)
    return np.add.reduceat(values, starts, axis=1) / counts


def main() -> None:
    args = parse_args()
    benchmarks = selected_benchmark_names(args.benchmark)
    out = args.out or args.results
    out.mkdir(parents=True, exist_ok=True)

    records = load_records(args.results, benchmarks, args.include_unfinished)
    examples = {
        (benchmark, example["id"]): example for benchmark in benchmarks for example in load_benchmark(benchmark)
    }

    scored = [int(value) for value in pair_table(args.vector_dir).index]

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    directions = load_directions(args.vector_dir, args.layer, scored, model.device)
    capture = Capture(model, args.layer)

    per_trace: list[dict[str, Any]] = []
    token_sum = np.zeros(len(scored), dtype=np.float64)
    token_square = np.zeros(len(scored), dtype=np.float64)
    token_count = 0
    benchmark_moments: dict[str, dict[str, Any]] = {}
    try:
        for index, record in enumerate(records, 1):
            if args.max_tokens and record["generated_token_count"] > args.max_tokens:
                print(f"{index}/{len(records)} {record['key']}: skipped, too long", flush=True)
                continue
            example = examples.get((record["benchmark"], record["id"]))
            if example is None:
                print(f"{index}/{len(records)} {record['key']}: question not in benchmark, skipped", flush=True)
                continue
            cosines = trace_cosines(
                model,
                tokenizer,
                capture,
                record,
                instruction(record["benchmark"], example["prompt"]),
                directions,
                args.chunk_tokens,
            )
            if cosines is None:
                print(f"{index}/{len(records)} {record['key']}: no usable reasoning span, skipped", flush=True)
                continue
            token_sum += cosines.sum(axis=1, dtype=np.float64)
            token_square += np.square(cosines, dtype=np.float64).sum(axis=1)
            token_count += cosines.shape[1]
            moments = benchmark_moments.setdefault(
                record["benchmark"],
                {
                    "sum": np.zeros(len(scored), dtype=np.float64),
                    "square": np.zeros(len(scored), dtype=np.float64),
                    "count": 0,
                },
            )
            moments["sum"] += cosines.sum(axis=1, dtype=np.float64)
            moments["square"] += np.square(cosines, dtype=np.float64).sum(axis=1)
            moments["count"] += cosines.shape[1]
            per_trace.append(
                {
                    "id": record["id"],
                    "benchmark": record["benchmark"],
                    "reasoning_status": record["reasoning_status"],
                    "correct": bool(record["correct"]),
                    "reasoning_tokens": cosines.shape[1],
                    "mean_cosine": cosines.mean(axis=1),
                    "curve": binned_all(cosines, args.bins),
                }
            )
            print(
                f"{index}/{len(records)} {record['key']} tokens={cosines.shape[1]} correct={record['correct']}",
                flush=True,
            )
    finally:
        capture.close()

    if not per_trace:
        raise SystemExit("No traces produced usable reasoning spans")

    token_mean = token_sum / token_count
    token_std = np.sqrt(np.maximum(token_square / token_count - token_mean**2, 0.0))
    token_std[token_std == 0] = 1.0
    benchmark_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, moments in benchmark_moments.items():
        mean = moments["sum"] / moments["count"]
        std = np.sqrt(np.maximum(moments["square"] / moments["count"] - mean**2, 0.0))
        std[std == 0] = 1.0
        benchmark_stats[name] = mean, std
    binned_z = np.stack(
        [
            (item["curve"] - benchmark_stats[item["benchmark"]][0][:, None])
            / benchmark_stats[item["benchmark"]][1][:, None]
            for item in per_trace
        ]
    ).astype(np.float16)

    suffix = "-include-unfinished" if args.include_unfinished else ""
    scores = out / f"concept-scores-L{args.layer}{suffix}.npz"
    np.savez_compressed(
        scores,
        pair_ids=np.asarray(scored, dtype=np.int32),
        ids=np.asarray([item["id"] for item in per_trace]),
        trace_benchmarks=np.asarray([item["benchmark"] for item in per_trace]),
        reasoning_status=np.asarray([item["reasoning_status"] for item in per_trace]),
        correct=np.asarray([item["correct"] for item in per_trace]),
        reasoning_tokens=np.asarray([item["reasoning_tokens"] for item in per_trace], dtype=np.int32),
        mean_cosine=np.stack([item["mean_cosine"] for item in per_trace]).astype(np.float32),
        binned_cosine=np.stack([item["curve"] for item in per_trace]).astype(np.float16),
        binned_z=binned_z,
        token_mean=token_mean.astype(np.float32),
        token_std=token_std.astype(np.float32),
        benchmark=",".join(benchmarks),
        benchmarks=np.asarray(benchmarks),
        layer=args.layer,
    )

    print(f"{len(per_trace)} traces -> {scores}")


if __name__ == "__main__":
    main()
