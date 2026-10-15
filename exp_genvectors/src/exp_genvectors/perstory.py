from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from exp_genvectors.artifacts import atomic_safetensors
from exp_genvectors.batching import Row, build_batch, mean_pool, token_budget_batches
from exp_genvectors.config import GENRES, LANGUAGES, MODEL_BY_CHECKPOINT, SKIP_TOKENS, STORY_MAX_TOKENS
from exp_genvectors.presentation import encode_wrapping

log = logging.getLogger("perstory")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_BY_CHECKPOINT))
    parser.add_argument("--presentation", default="assistant")
    parser.add_argument("--dataset", type=Path, default=Path("data/semantic.parquet"))
    parser.add_argument("--pairs", required=True, type=Path, help="json file with {'all': [pair ids]}")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--token-budget", type=int, default=32768)
    parser.add_argument("--split-model", action="store_true", help="pipeline the model across every visible GPU")
    parser.add_argument("--buckets", type=int, nargs="*", default=None, help="restrict to these buckets")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    wanted = sorted(json.loads(args.pairs.read_text())["all"])
    import pyarrow.parquet as pq

    table = pq.read_table(args.dataset).to_pydict()
    rows_by_pair: dict[int, list[dict]] = {p: [] for p in wanted}
    for i in range(len(table["pair_id"])):
        pid = table["pair_id"][i]
        if pid in rows_by_pair:
            rows_by_pair[pid].append({k: table[k][i] for k in table})
    log.info("selected %s pairs, %s rows", len(wanted), sum(len(v) for v in rows_by_pair.values()))

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    blocks, hidden = int(config.num_hidden_layers), int(config.hidden_size)
    layers = blocks + 1
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    attention = "eager" if config.model_type == "gemma2" else "sdpa"
    model = AutoModel.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation=attention,
        local_files_only=True,
        device_map="auto" if args.split_model else {"": "cuda"},
    ).eval()
    core = getattr(model, "model", model)
    store_device = torch.device("cuda:0")
    input_device = core.embed_tokens.weight.device
    weights_holder: list[torch.Tensor] = [torch.empty(0)]
    captured: list[torch.Tensor | None] = [None] * layers

    def hook(position: int, _m, _i, output) -> None:
        state = output[0] if isinstance(output, tuple) else output
        pooled = mean_pool(state, weights_holder[0].to(state.device, non_blocking=True))
        captured[position] = pooled.to(store_device, non_blocking=True)

    core.embed_tokens.register_forward_hook(partial(hook, 0))
    for position, block in enumerate(core.layers):
        block.register_forward_hook(partial(hook, position + 1))

    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    args.out.mkdir(parents=True, exist_ok=True)
    for pid in wanted:
        records = rows_by_pair[pid]
        if args.buckets:
            records = [r for r in records if int(r["bucket"]) in set(args.buckets)]
        concept, antagonist = records[0]["concept"], records[0]["antagonist"]
        encoded = tokenizer(
            [r["concept_text"] for r in records] + [r["antagonist_text"] for r in records],
            add_special_tokens=False,
            truncation=True,
            max_length=STORY_MAX_TOKENS,
        )["input_ids"]
        n = len(records)
        wrap_cache: dict[tuple[int, int, int], object] = {}

        def wrapping(row: Row, side: int):
            key = (side, row.language, row.genre)
            if key not in wrap_cache:
                wrap_cache[key] = encode_wrapping(
                    tokenizer,
                    args.presentation,
                    (concept, antagonist)[side],
                    LANGUAGES[row.language],
                    GENRES[row.genre],
                )
            return wrap_cache[key]

        row_objs = [
            Row(
                pair=0,
                bucket=int(r["bucket"]),
                genre=int(r["genre_id"]),
                language=int(r["language_id"]),
                generator=int(r["generator_id"]),
                concept_ids=tuple(encoded[i]),
                antagonist_ids=tuple(encoded[n + i]),
            )
            for i, r in enumerate(records)
        ]
        lengths = [max(len(r.concept_ids), len(r.antagonist_ids)) for r in row_objs]
        batches = token_budget_batches(lengths, args.token_budget)
        diff = np.empty((n, layers, hidden), dtype=np.float16)
        with torch.inference_mode():
            for batch_positions in batches:
                batch = build_batch(
                    [row_objs[i] for i in batch_positions], wrapping, pad_token_id=pad, skip_tokens=SKIP_TOKENS
                )
                weights_holder[0] = batch.pooling_weights.to(input_device, non_blocking=True)
                captured[:] = [None] * layers
                model(
                    input_ids=batch.input_ids.to(input_device, non_blocking=True),
                    attention_mask=batch.attention_mask.to(input_device, non_blocking=True),
                    use_cache=False,
                )
                pooled = torch.stack([c for c in captured]).view(layers, 2, len(batch_positions), hidden)
                d = (pooled[:, 0] - pooled[:, 1]).permute(1, 0, 2).cpu().numpy().astype(np.float16)
                for k, position in enumerate(batch_positions):
                    diff[position] = d[k]
        metadata = {
            "manifest": json.dumps(
                {
                    "model": args.model,
                    "presentation": args.presentation,
                    "pair_id": pid,
                    "concept": concept,
                    "antagonist": antagonist,
                    "axes": ["row", "layer", "hidden"],
                    "rows": [
                        {
                            "row_id": r["row_id"],
                            "genre_id": int(r["genre_id"]),
                            "language_id": int(r["language_id"]),
                            "generator_id": int(r["generator_id"]),
                            "bucket": int(r["bucket"]),
                        }
                        for r in records
                    ],
                },
                sort_keys=True,
            )
        }
        atomic_safetensors({"diff": diff}, args.out / f"pair-{pid:04d}.safetensors", metadata)
        log.info("pair %s (%s vs %s): %s rows written", pid, concept, antagonist, n)


if __name__ == "__main__":
    main()
