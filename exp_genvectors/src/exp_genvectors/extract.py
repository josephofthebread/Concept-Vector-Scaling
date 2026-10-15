from __future__ import annotations

import json
import logging
import os
import time
from argparse import ArgumentParser, BooleanOptionalAction
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer

from exp_genvectors.artifacts import CaptureManifest, atomic_safetensors, pack_capture, sha256_file
from exp_genvectors.batching import Batch, Row, build_batch, mean_pool, token_budget_batches
from exp_genvectors.config import (
    BUCKETS,
    GENRES,
    LANGUAGES,
    MODEL_BY_CHECKPOINT,
    PRESENTATIONS,
    SKIP_TOKENS,
    STORY_MAX_TOKENS,
)
from exp_genvectors.presentation import Wrapping, allowed_presentations, encode_wrapping
from exp_genvectors.stats import SufficientStatistics

log = logging.getLogger("extract")


def tokenize_batch(batch: dict[str, list[Any]], tokenizer: Any) -> dict[str, list[Any]]:
    texts = [str(text) for text in batch["concept_text"]] + [str(text) for text in batch["antagonist_text"]]
    encoded = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=STORY_MAX_TOKENS)
    size = len(batch["concept_text"])
    return {
        "concept_ids": encoded["input_ids"][:size],
        "antagonist_ids": encoded["input_ids"][size:],
        "length": [
            max(len(a), len(b)) for a, b in zip(encoded["input_ids"][:size], encoded["input_ids"][size:], strict=True)
        ],
        "concept_hit": [
            int(str(name).casefold() in str(text).casefold())
            for name, text in zip(batch["concept"], batch["concept_text"], strict=True)
        ],
        "antagonist_hit": [
            int(str(name).casefold() in str(text).casefold())
            for name, text in zip(batch["antagonist"], batch["antagonist_text"], strict=True)
        ],
    }


class WrappingCache:
    """Wrappings only vary over (pair, side, language, genre), so encode each once."""

    def __init__(
        self, tokenizer: Any, presentation: str, poles: list[tuple[str, str]], pairs_meta: list[dict] | None = None
    ) -> None:
        self.tokenizer, self.presentation, self.poles = tokenizer, presentation, poles
        self.pairs_meta = pairs_meta
        self.cache: dict[tuple[int, int, int, int], Wrapping] = {}

    def __call__(self, row: Row, side: int) -> Wrapping:
        key = (row.pair, side, row.language, row.genre)
        if key not in self.cache:
            self.cache[key] = encode_wrapping(
                self.tokenizer,
                self.presentation,
                self.poles[row.pair][side],
                LANGUAGES[row.language],
                GENRES[row.genre],
                self.pairs_meta[row.pair] if self.pairs_meta is not None else None,
            )
        return self.cache[key]


def collate(records: list[dict[str, Any]], wrapping: WrappingCache, pad: int) -> Batch:
    return build_batch(
        [
            Row(
                pair=int(record["pair_id"]),
                bucket=int(record["bucket"]),
                genre=int(record["genre_id"]),
                language=int(record["language_id"]),
                generator=int(record["generator_id"]),
                concept_ids=tuple(record["concept_ids"]),
                antagonist_ids=tuple(record["antagonist_ids"]),
                concept_hit=int(record["concept_hit"]),
                antagonist_hit=int(record["antagonist_hit"]),
            )
            for record in records
        ],
        wrapping,
        pad_token_id=pad,
        skip_tokens=SKIP_TOKENS,
    )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_BY_CHECKPOINT))
    parser.add_argument("--presentation", required=True, choices=PRESENTATIONS)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--token-budget", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 8))
    parser.add_argument("--max-rows", type=int, default=0, help="bounded smoke run; never a production artifact")
    parser.add_argument("--buckets", type=int, nargs="*", default=None, help="held-out subset: keep only these buckets")
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    parser.add_argument("--split-model", action="store_true", help="pipeline the model across every visible GPU")
    parser.add_argument("--max-gpu-memory", help="per-GPU cap, e.g. 70GiB; enables CPU offload of the remainder")
    parser.add_argument("--cpu-memory", default="600GiB", help="host RAM budget for offloaded layers")
    parser.add_argument("--minimal", action="store_true", help="skip second moments and genre/generator cells")
    parser.add_argument("--pair-subset", type=Path, help="json list of pair_ids to keep")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    spec = MODEL_BY_CHECKPOINT[args.model]
    if args.presentation not in allowed_presentations(spec.instruction_tuned):
        raise SystemExit(f"{args.presentation!r} is not defined for {args.model}")
    config = AutoConfig.from_pretrained(args.model, local_files_only=args.offline)
    blocks, hidden = int(config.num_hidden_layers), int(config.hidden_size)
    layers = blocks + 1
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.offline, use_fast=True)

    from datasets import load_dataset

    data = load_dataset("parquet", data_files=str(args.dataset), split="train")
    subset: list[int] | None = None
    if args.pair_subset:
        subset = sorted(json.loads(args.pair_subset.read_text()))
        index = {old: new for new, old in enumerate(subset)}
        keep = set(subset)
        data = data.filter(lambda row: row["pair_id"] in keep, desc="pair subset")
        data = data.map(lambda row: {"pair_id": index[row["pair_id"]]}, desc="reindex pairs")
    if args.buckets:
        kept = set(args.buckets)
        data = data.filter(lambda row: row["bucket"] in kept, desc="bucket filter")
    data = data.shard(num_shards=args.shards, index=args.shard, contiguous=False)
    if args.max_rows:
        data = data.select(range(min(args.max_rows, len(data))))
    data = data.map(
        partial(tokenize_batch, tokenizer=tokenizer),
        batched=True,
        batch_size=512,
        num_proc=args.workers or None,
        desc="tokenizing",
    )
    batches = token_budget_batches(list(data["length"]), args.token_budget)

    pairs_meta = json.loads(args.ontology.read_text())["pairs"]
    pairs_meta.sort(key=lambda pair: int(pair["pair_id"]))
    if subset is not None:
        pairs_meta = [pairs_meta[old] for old in subset]
    poles = [(str(pair["concept"]), str(pair["antagonist"])) for pair in pairs_meta]
    wrapping = WrappingCache(tokenizer, args.presentation, poles, pairs_meta)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    loader: DataLoader[Any] = DataLoader(
        data,
        batch_sampler=batches,
        collate_fn=partial(collate, wrapping=wrapping, pad=int(pad)),
        num_workers=args.workers,
        pin_memory=True,
    )

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    torch.backends.cuda.enable_cudnn_sdp(False)
    attention = "eager" if config.model_type == "gemma2" else "sdpa"
    device_map = "auto" if args.split_model else {"": "cuda"}
    limits = None
    if args.max_gpu_memory:
        limits = {i: args.max_gpu_memory for i in range(torch.cuda.device_count())}
        limits["cpu"] = args.cpu_memory
    model = AutoModel.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation=attention,
        local_files_only=args.offline,
        device_map=device_map,
        max_memory=limits,
    ).eval()
    core = getattr(model, "model", model)
    if len(core.layers) != blocks:
        raise SystemExit(f"config declares {blocks} blocks but model exposes {len(core.layers)}")
    log.info(
        "%s: %s blocks, hidden %s, %s, %s attention, %s batches",
        args.model,
        blocks,
        hidden,
        dtype,
        attention,
        len(batches),
    )

    accumulator_device = torch.device("cuda:0")
    input_device = core.embed_tokens.weight.device
    weights_holder: list[torch.Tensor] = [torch.empty(0)]
    captured: list[torch.Tensor | None] = [None] * layers

    def hook(position: int, _module: Any, _inputs: Any, output: Any) -> None:
        state = output[0] if isinstance(output, tuple) else output
        pooled = mean_pool(state, weights_holder[0].to(state.device, non_blocking=True))
        captured[position] = pooled.to(accumulator_device, non_blocking=True)

    core.embed_tokens.register_forward_hook(partial(hook, 0))
    for position, block in enumerate(core.layers):
        block.register_forward_hook(partial(hook, position + 1))

    stats = SufficientStatistics(
        layers=layers,
        pairs=len(poles),
        hidden=hidden,
        buckets=BUCKETS,
        device=accumulator_device,
        moments=not args.minimal,
        cells=not args.minimal,
    )
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        print(
            f"cuda:{index} {(total - free) / 2**30:.1f} GiB used of {total / 2**30:.1f} GiB "
            f"after weights + accumulators",
            flush=True,
        )
    started = time.monotonic()
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            weights_holder[0] = batch.pooling_weights.to(input_device, non_blocking=True)
            captured[:] = [None] * layers
            model(
                input_ids=batch.input_ids.to(input_device, non_blocking=True),
                attention_mask=batch.attention_mask.to(input_device, non_blocking=True),
                use_cache=False,
            )
            missing = [layer for layer, value in enumerate(captured) if value is None]
            if missing:
                raise RuntimeError(f"forward hooks missed layers {missing}")
            rows = batch.pair.numel()
            pooled = torch.stack(captured).view(layers, 2, rows, hidden)  # type: ignore[arg-type]
            stats.add(pooled, batch)
            if step % 50 == 0 or step == len(batches):
                rate = step / (time.monotonic() - started)
                log.info("batch %s/%s, %.1f h left", step, len(batches), (len(batches) - step) / rate / 3600)

    manifest = CaptureManifest(
        model=args.model,
        presentation=args.presentation,
        dataset=args.dataset.stem,
        dataset_sha256=sha256_file(args.dataset),
        shard=args.shard,
        shards=args.shards,
        layers=layers,
        hidden_size=hidden,
        pairs=len(poles),
        buckets=BUCKETS,
        dtype=str(dtype).removeprefix("torch."),
        skip_tokens=SKIP_TOKENS,
    )
    tensors = stats.numpy()
    summary = {
        "paired_rows": int(tensors["counts"].sum()) // 2,
        "tokens": int(tensors["tokens"].sum()),
        "nonfinite_rows": stats.nonfinite_rows,
    }
    atomic_safetensors(
        pack_capture(tensors, mean16=args.dataset.stem != "syntax"), args.out, manifest.metadata(summary)
    )
    log.info("wrote %s (%.2f GiB), %s", args.out, args.out.stat().st_size / 2**30, summary)


if __name__ == "__main__":
    main()
