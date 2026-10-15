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

log = logging.getLogger("langscores")
MANDARIN = LANGUAGES.index("Mandarin Chinese")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_BY_CHECKPOINT))
    parser.add_argument("--presentation", default="assistant")
    parser.add_argument("--dataset", type=Path, default=Path("data/semantic.parquet"))
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--token-budget", type=int, default=32768)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from safetensors import safe_open

    with safe_open(str(args.capture), framework="np") as f:
        lang_count = f.get_tensor("language_count").astype(np.float32)
        lang_sum = f.get_tensor("language_sum").astype(np.float32)
        gen_sum = f.get_tensor("generator_sum").astype(np.float32)
    lang_mu = lang_sum
    gen_mu = gen_sum
    v_lang = lang_mu[:, :, 0] - lang_mu[:, :, 1]
    weights = lang_count[:, 0]
    others = [i for i in range(v_lang.shape[2]) if i != MANDARIN]
    rest_centroid = np.average(v_lang[:, :, others], axis=2, weights=weights[:, others].mean(0))
    mandarin_dir = v_lang[:, :, MANDARIN] - rest_centroid
    v_gen = gen_mu[:, :, 0] - gen_mu[:, :, 1]
    generator_dir = v_gen[:, :, 1] - v_gen[:, :, 0]
    log.info("directions ready: %s", mandarin_dir.shape)

    import pyarrow.parquet as pq

    table = pq.read_table(args.dataset).to_pydict()
    n_pairs = mandarin_dir.shape[1]
    rows_by_pair: dict[int, list[dict]] = {p: [] for p in range(n_pairs)}
    for i in range(len(table["pair_id"])):
        rows_by_pair[table["pair_id"][i]].append({k: table[k][i] for k in table})

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    blocks, hidden = int(config.num_hidden_layers), int(config.hidden_size)
    layers = blocks + 1
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    attention = "eager" if config.model_type == "gemma2" else "sdpa"
    model = AutoModel.from_pretrained(
        args.model, dtype=dtype, attn_implementation=attention, local_files_only=True, device_map={"": "cuda"}
    ).eval()
    core = getattr(model, "model", model)
    weights_holder: list[torch.Tensor] = [torch.empty(0)]
    captured: list[torch.Tensor | None] = [None] * layers

    def hook(position: int, _m, _i, output) -> None:
        state = output[0] if isinstance(output, tuple) else output
        captured[position] = mean_pool(state, weights_holder[0])

    core.embed_tokens.register_forward_hook(partial(hook, 0))
    for position, block in enumerate(core.layers):
        block.register_forward_hook(partial(hook, position + 1))

    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    args.out.mkdir(parents=True, exist_ok=True)
    for pid in range(n_pairs):
        done_path = args.out / f"pair-{pid:04d}.safetensors"
        if done_path.exists():
            continue
        records = rows_by_pair[pid]
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
        m_dir = torch.from_numpy(mandarin_dir[:, pid]).cuda()
        centroid = torch.from_numpy(rest_centroid[:, pid]).cuda()
        g_dir = torch.from_numpy(generator_dir[:, pid]).cuda()
        scores = np.empty((n, layers, 3), dtype=np.float32)
        lengths = [max(len(r.concept_ids), len(r.antagonist_ids)) for r in row_objs]
        with torch.inference_mode():
            for batch_positions in token_budget_batches(lengths, args.token_budget):
                batch = build_batch(
                    [row_objs[i] for i in batch_positions], wrapping, pad_token_id=pad, skip_tokens=SKIP_TOKENS
                )
                weights_holder[0] = batch.pooling_weights.cuda(non_blocking=True)
                captured[:] = [None] * layers
                model(
                    input_ids=batch.input_ids.cuda(non_blocking=True),
                    attention_mask=batch.attention_mask.cuda(non_blocking=True),
                    use_cache=False,
                )
                pooled = torch.stack([c for c in captured]).view(layers, 2, len(batch_positions), hidden)
                x = pooled[:, 0] - pooled[:, 1]
                s0 = torch.einsum("lrh,lh->lr", x, m_dir)
                s1 = (x - centroid[:, None]).norm(dim=2)
                s2 = torch.einsum("lrh,lh->lr", x, g_dir)
                stacked = torch.stack([s0, s1, s2], dim=2).permute(1, 0, 2).cpu().numpy()
                for k, position in enumerate(batch_positions):
                    scores[position] = stacked[k]
        metadata = {
            "manifest": json.dumps(
                {
                    "model": args.model,
                    "presentation": args.presentation,
                    "pair_id": pid,
                    "concept": concept,
                    "antagonist": antagonist,
                    "axes": ["row", "layer", "score"],
                    "scores": ["mandarin_proj", "rest_distance", "generator_proj"],
                    "rows": [
                        {
                            "language_id": int(r["language_id"]),
                            "generator_id": int(r["generator_id"]),
                            "genre_id": int(r["genre_id"]),
                        }
                        for r in records
                    ],
                },
                sort_keys=True,
            )
        }
        atomic_safetensors({"scores": scores}, done_path, metadata)
        if pid % 25 == 0:
            log.info("pair %s/%s done", pid, n_pairs)


if __name__ == "__main__":
    main()
