from __future__ import annotations

import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from exp_gendata.batching import Row, build_batch, mean_pool, token_budget_batches
from exp_gendata.config import GENRES, LANGUAGES, MODEL_BY_CHECKPOINT, SKIP_TOKENS, STORY_MAX_TOKENS
from exp_gendata.presentation import encode_wrapping

log = logging.getLogger("convcurve")

GRID = (1, 2, 4, 8, 16, 32, 64, 128)
TEST_PER_POLE = 25
REPEATS = 8


def auroc(y: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC; scores have no ties in practice (fp32 dot products)."""
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[np.argsort(scores, kind="stable")] = np.arange(1, len(scores) + 1)
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_BY_CHECKPOINT))
    parser.add_argument("--presentation", default="assistant")
    parser.add_argument("--dataset", type=Path, default=Path("data/semantic.parquet"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument(
        "--cap-per-lang", type=int, default=70, help="max stories per (pole, language); 5 langs -> ~350/pole"
    )
    parser.add_argument("--token-budget", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    spec = MODEL_BY_CHECKPOINT[args.model]
    if not spec.instruction_tuned:
        raise SystemExit(f"{args.model} has no chat template; assistant vectors need an instruction-tuned model")

    tag = args.model.split("/")[-1]
    out = Path(f"work/convcurve-{tag}")
    out.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(
        args.dataset,
        columns=[
            "pair_id",
            "language_id",
            "genre_id",
            "generator_id",
            "concept",
            "antagonist",
            "concept_text",
            "antagonist_text",
            "bucket",
        ],
    ).to_pydict()
    rows_by_pair: dict[int, list[int]] = {}
    for i, pid in enumerate(table["pair_id"]):
        rows_by_pair.setdefault(int(pid), []).append(i)
    pairs = sorted(rows_by_pair)[args.shard :: args.shards]
    log.info("%s shard %s/%s: %s pairs", args.model, args.shard, args.shards, len(pairs))

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
        device_map={"": "cuda"},
    ).eval()
    core = getattr(model, "model", model)
    input_device = core.embed_tokens.weight.device
    store_device = torch.device("cuda:0")
    weights_holder: list[torch.Tensor] = [torch.empty(0)]
    captured: list[torch.Tensor | None] = [None] * layers

    def hook(position: int, _m, _i, output) -> None:
        state = output[0] if isinstance(output, tuple) else output
        captured[position] = mean_pool(state, weights_holder[0].to(state.device, non_blocking=True)).to(store_device)

    core.embed_tokens.register_forward_hook(partial(hook, 0))
    for position, block in enumerate(core.layers):
        block.register_forward_hook(partial(hook, position + 1))

    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    band = np.arange(layers)
    curve_rng = np.random.default_rng(args.seed)
    sample_rng = np.random.default_rng(args.seed + 1)

    for pi, pid in enumerate(pairs):
        dst = out / f"pair-{pid:04d}.npz"
        if dst.exists():
            continue
        records = [{k: table[k][i] for k in table} for i in rows_by_pair[pid]]
        by_lang: dict[int, list[dict]] = {}
        for r in records:
            by_lang.setdefault(int(r["language_id"]), []).append(r)
        chosen: list[dict] = []
        for idxs in by_lang.values():
            take = (
                idxs
                if len(idxs) <= args.cap_per_lang
                else [idxs[j] for j in sample_rng.choice(len(idxs), args.cap_per_lang, replace=False)]
            )
            chosen.extend(take)
        concept_name, antagonist_name = chosen[0]["concept"], chosen[0]["antagonist"]
        encoded = tokenizer(
            [r["concept_text"] for r in chosen] + [r["antagonist_text"] for r in chosen],
            add_special_tokens=False,
            truncation=True,
            max_length=STORY_MAX_TOKENS,
        )["input_ids"]
        n = len(chosen)
        wrap_cache: dict[tuple[int, int, int], object] = {}

        def wrapping(row: Row, side: int):
            key = (side, row.language, row.genre)
            if key not in wrap_cache:
                wrap_cache[key] = encode_wrapping(
                    tokenizer,
                    args.presentation,
                    (concept_name, antagonist_name)[side],
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
            for i, r in enumerate(chosen)
        ]
        lengths = [max(len(r.concept_ids), len(r.antagonist_ids)) for r in row_objs]
        concept_reps = np.empty((n, layers, hidden), dtype=np.float32)
        antagonist_reps = np.empty((n, layers, hidden), dtype=np.float32)
        with torch.inference_mode():
            for batch_positions in token_budget_batches(lengths, args.token_budget):
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
                pooled = pooled.permute(1, 2, 0, 3).float().cpu().numpy()
                for k, position in enumerate(batch_positions):
                    concept_reps[position] = pooled[0, k]
                    antagonist_reps[position] = pooled[1, k]
        n_layers = concept_reps.shape[1]
        auroc_curve = np.full((n_layers, len(GRID)), np.nan, dtype=np.float32)
        conv_curve = np.full((n_layers, len(GRID)), np.nan, dtype=np.float32)
        nc, na = len(concept_reps), len(antagonist_reps)

        for li in range(n_layers):
            rc, ra = concept_reps[:, li, :], antagonist_reps[:, li, :]
            pc, pa = curve_rng.permutation(nc), curve_rng.permutation(na)
            test = np.concatenate([rc[pc[:TEST_PER_POLE]], ra[pa[:TEST_PER_POLE]]])
            y_test = np.concatenate([np.ones(TEST_PER_POLE), np.zeros(TEST_PER_POLE)])
            train_c, train_a = pc[TEST_PER_POLE:], pa[TEST_PER_POLE:]
            for gi, N in enumerate(GRID):
                if N > len(train_c) or N > len(train_a):
                    break
                vals = []
                for _ in range(REPEATS):
                    diff = rc[curve_rng.choice(train_c, N, replace=False)].mean(0) - ra[
                        curve_rng.choice(train_a, N, replace=False)
                    ].mean(0)
                    vals.append(auroc(y_test, test @ diff))
                auroc_curve[li, gi] = float(np.mean(vals))
            for gi, N in enumerate(GRID):
                if 2 * N > nc or 2 * N > na:
                    break
                cosines = []
                for _ in range(REPEATS):
                    sc, sa = curve_rng.permutation(nc)[: 2 * N], curve_rng.permutation(na)[: 2 * N]
                    va = rc[sc[:N]].mean(0) - ra[sa[:N]].mean(0)
                    vb = rc[sc[N:]].mean(0) - ra[sa[N:]].mean(0)
                    denom = np.linalg.norm(va) * np.linalg.norm(vb)
                    cosines.append(float(va @ vb / denom) if denom > 0 else np.nan)
                conv_curve[li, gi] = float(np.nanmean(cosines))
        np.savez_compressed(
            dst,
            auroc=auroc_curve,
            conv=conv_curve,
            layers=band.astype(np.int32),
            grid=np.array(GRID, np.int32),
            n_stories=np.int32(n),
        )
        if pi % 10 == 0:
            log.info(
                "  %s/%s pair %s: %s stories/pole, best-layer plateau AUROC %.3f / conv %.3f",
                pi,
                len(pairs),
                pid,
                n,
                np.nanmax(auroc_curve[:, -1]),
                np.nanmax(conv_curve[:, -1]),
            )
    log.info("=== shard %s done ===", args.shard)


if __name__ == "__main__":
    main()
