from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from functools import partial
from itertools import combinations
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from exp_gendata.batching import Row, build_batch, mean_pool, token_budget_batches
from exp_gendata.config import GENRES, LANGUAGES, MODEL_BY_CHECKPOINT, SKIP_TOKENS, STORY_MAX_TOKENS
from exp_gendata.presentation import encode_wrapping

log = logging.getLogger("convcurve2")

POWERS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
TEST_ROWS = 50
REPEATS = 8
MATCHED_SIZE = 200
LANG_REPEATS = 4
FORMATS = ("raw", "user", "assistant", "paired_assistant")
LANG_COMBOS = [c for k in range(1, len(LANGUAGES)) for c in combinations(range(len(LANGUAGES)), k)]


def auroc_rows(scores: np.ndarray, n_pos: int) -> np.ndarray:
    """scores [layer, test] with the first n_pos columns positive -> AUROC per layer (no ties in fp32)."""
    ranks = scores.argsort(axis=1).argsort(axis=1) + 1.0
    n_neg = scores.shape[1] - n_pos
    return (ranks[:, :n_pos].sum(axis=1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def mean_diffs(delta: np.ndarray, index_lists: list[np.ndarray]) -> np.ndarray:
    """Matched rows make every subset vector a mean of per-row differences: one GEMM for all subsets.
    delta [row, layer, hidden] -> [subset, layer, hidden]."""
    n, n_layers, hidden = delta.shape
    weights = np.zeros((len(index_lists), n), dtype=np.float32)
    for i, idx in enumerate(index_lists):
        weights[i, idx] = 1.0 / len(idx)
    return (weights @ delta.reshape(n, -1)).reshape(len(index_lists), n_layers, hidden)


def score_many(concept: np.ndarray, antagonist: np.ndarray, rows: np.ndarray, diffs: np.ndarray) -> np.ndarray:
    """AUROC of each diff on the stories of `rows` (concept positive) -> [layer, subset]."""
    reps = np.concatenate([concept[rows], antagonist[rows]]).transpose(1, 0, 2)
    scores = reps @ diffs.transpose(1, 2, 0)
    ranks = scores.argsort(axis=1).argsort(axis=1) + 1.0
    n_pos, n_neg = len(rows), len(rows)
    return (ranks[:, :n_pos].sum(axis=1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_BY_CHECKPOINT))
    parser.add_argument("--formats", nargs="+", default=list(FORMATS))
    parser.add_argument("--dataset", type=Path, default=Path("data/semantic.parquet"))
    parser.add_argument("--ontology", type=Path, default=Path("data/semantic.ontology.json"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--token-budget", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not MODEL_BY_CHECKPOINT[args.model].instruction_tuned:
        raise SystemExit(f"{args.model}: chat formats need an instruction-tuned model")
    tag = args.model.split("/")[-1]
    out = Path(f"work/convcurve2-{tag}")
    out.mkdir(parents=True, exist_ok=True)

    meta = {int(p["pair_id"]): p for p in json.loads(args.ontology.read_text())["pairs"]}
    table = pq.read_table(
        args.dataset,
        columns=["pair_id", "language_id", "genre_id", "generator_id", "concept_text", "antagonist_text", "bucket"],
    ).to_pydict()
    rows_by_pair: dict[int, list[int]] = {}
    for i, pid in enumerate(table["pair_id"]):
        rows_by_pair.setdefault(int(pid), []).append(i)
    pairs = sorted(rows_by_pair)[args.shard :: args.shards]
    n_generators = int(max(table["generator_id"])) + 1
    log.info("%s shard %s/%s: %s pairs, formats %s", args.model, args.shard, args.shards, len(pairs), args.formats)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    layers, hidden = int(config.num_hidden_layers) + 1, int(config.hidden_size)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    attention = "eager" if config.model_type == "gemma2" else "sdpa"
    model = AutoModel.from_pretrained(
        args.model, dtype=dtype, attn_implementation=attention, local_files_only=True, device_map={"": "cuda"}
    ).eval()
    core = getattr(model, "model", model)
    input_device = core.embed_tokens.weight.device
    weights_holder: list[torch.Tensor] = [torch.empty(0)]
    captured: list[torch.Tensor | None] = [None] * layers

    def hook(position: int, _m, _i, output) -> None:
        state = output[0] if isinstance(output, tuple) else output
        captured[position] = mean_pool(state, weights_holder[0].to(state.device, non_blocking=True))

    core.embed_tokens.register_forward_hook(partial(hook, 0))
    for position, block in enumerate(core.layers):
        block.register_forward_hook(partial(hook, position + 1))
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

    for pi, pid in enumerate(pairs):
        dst = out / f"pair-{pid:04d}.npz"
        if dst.exists():
            continue
        rng = np.random.default_rng([args.seed, pid])
        chosen = rows_by_pair[pid]
        n = len(chosen)
        groups = {
            field: np.array([int(table[f"{field}_id"][i]) for i in chosen])
            for field in ("language", "genre", "generator")
        }
        order = rng.permutation(n)
        test, train = order[:TEST_ROWS], order[TEST_ROWS:]
        auc_grid = [N for N in POWERS if N < len(train)] + [len(train)]
        conv_grid = [N for N in POWERS if 2 * N <= n] + [n // 2]
        draws: dict = {
            "test": test,
            "auc_grid": auc_grid,
            "conv_grid": conv_grid,
            "train": [[rng.choice(train, N, replace=False) for _ in range(REPEATS)] for N in auc_grid],
            "disjoint": [],
        }
        for N in conv_grid:
            disjoint_pairs = []
            for _ in range(REPEATS):
                pick = rng.permutation(n)[: 2 * N]
                disjoint_pairs.append((pick[:N], pick[N:]))
            draws["disjoint"].append(disjoint_pairs)
        by_lang = {g: np.where(groups["language"] == g)[0] for g in range(len(LANGUAGES))}
        draws["by_lang"] = by_lang
        draws["lang_size"] = []
        for combo in LANG_COMBOS:
            reps = []
            for _ in range(LANG_REPEATS):
                share = np.array_split(np.arange(MATCHED_SIZE), len(combo))
                parts = [
                    rng.choice(by_lang[g], min(len(s), len(by_lang[g])), replace=False) for g, s in zip(combo, share)
                ]
                reps.append(np.concatenate(parts))
            draws["lang_size"].append(reps)
        encoded = tokenizer(
            [table["concept_text"][i] for i in chosen] + [table["antagonist_text"][i] for i in chosen],
            add_special_tokens=False,
            truncation=True,
            max_length=STORY_MAX_TOKENS,
        )["input_ids"]
        row_objs = [
            Row(
                pair=0,
                bucket=int(table["bucket"][i]),
                genre=int(table["genre_id"][i]),
                language=int(table["language_id"][i]),
                generator=int(table["generator_id"][i]),
                concept_ids=tuple(encoded[k]),
                antagonist_ids=tuple(encoded[n + k]),
            )
            for k, i in enumerate(chosen)
        ]
        lengths = [max(len(r.concept_ids), len(r.antagonist_ids)) for r in row_objs]
        pair_meta = meta[pid]
        poles = (str(pair_meta["concept"]), str(pair_meta["antagonist"]))
        result: dict[str, np.ndarray] = {}
        for fmt in args.formats:
            cache: dict[tuple[int, int, int], object] = {}

            def wrapping(row: Row, side: int, fmt: str = fmt, cache: dict = cache):
                key = (side, row.language, row.genre)
                if key not in cache:
                    cache[key] = encode_wrapping(
                        tokenizer, fmt, poles[side], LANGUAGES[row.language], GENRES[row.genre], pair_meta
                    )
                return cache[key]

            concept = np.empty((n, layers, hidden), dtype=np.float32)
            antagonist = np.empty((n, layers, hidden), dtype=np.float32)
            with torch.inference_mode():
                for positions in token_budget_batches(lengths, args.token_budget):
                    batch = build_batch(
                        [row_objs[k] for k in positions], wrapping, pad_token_id=pad, skip_tokens=SKIP_TOKENS
                    )
                    weights_holder[0] = batch.pooling_weights.to(input_device, non_blocking=True)
                    captured[:] = [None] * layers
                    model(
                        input_ids=batch.input_ids.to(input_device, non_blocking=True),
                        attention_mask=batch.attention_mask.to(input_device, non_blocking=True),
                        use_cache=False,
                    )
                    pooled = torch.stack([c.to(input_device) for c in captured]).view(layers, 2, len(positions), hidden)
                    pooled = pooled.permute(1, 2, 0, 3).float().cpu().numpy()
                    concept[positions] = pooled[0]
                    antagonist[positions] = pooled[1]
            n_layers = concept.shape[1]
            delta = concept - antagonist
            measured: dict[str, np.ndarray] = {}

            grid, reps = len(draws["auc_grid"]), REPEATS
            diffs = mean_diffs(delta, [idx for per_n in draws["train"] for idx in per_n])
            auc = score_many(concept, antagonist, draws["test"], diffs)
            measured["auroc"] = auc.reshape(n_layers, grid, reps).mean(-1).astype(np.float32)

            firsts = mean_diffs(delta, [a for per_n in draws["disjoint"] for a, _ in per_n])
            seconds = mean_diffs(delta, [b for per_n in draws["disjoint"] for _, b in per_n])
            cos = np.einsum("dlh,dlh->dl", firsts, seconds) / np.maximum(
                np.linalg.norm(firsts, axis=2) * np.linalg.norm(seconds, axis=2), 1e-30
            )
            measured["conv"] = cos.reshape(len(draws["conv_grid"]), reps, n_layers).mean(1).T.astype(np.float32)
            del diffs, firsts, seconds

            by_lang = draws["by_lang"]
            n_combo = len(LANG_COMBOS)
            lists = [np.concatenate([by_lang[g] for g in combo]) for combo in LANG_COMBOS]
            lists += [idx for per_combo in draws["lang_size"] for idx in per_combo]
            diffs = mean_diffs(delta, lists)
            lang_all = np.full((n_layers, n_combo, len(LANGUAGES)), np.nan, dtype=np.float32)
            lang_size = np.full_like(lang_all, np.nan)
            for g in range(len(LANGUAGES)):
                if len(by_lang[g]) < 8:
                    continue
                auc = score_many(concept, antagonist, by_lang[g], diffs)
                for ci, combo in enumerate(LANG_COMBOS):
                    if g in combo:
                        continue
                    lang_all[:, ci, g] = auc[:, ci]
                    start = n_combo + ci * LANG_REPEATS
                    lang_size[:, ci, g] = auc[:, start : start + LANG_REPEATS].mean(-1)
            measured["lang_all"], measured["lang_size"] = lang_all, lang_size
            del diffs

            for field, size in (("genre", len(GENRES)), ("generator", n_generators)):
                arr = np.full((n_layers, size), np.nan, dtype=np.float32)
                live = [g for g in range(size) if (groups[field] == g).sum() >= 8 and (groups[field] != g).sum() >= 8]
                if live:
                    diffs = mean_diffs(delta, [np.where(groups[field] != g)[0] for g in live])
                    for k, g in enumerate(live):
                        arr[:, g] = score_many(concept, antagonist, np.where(groups[field] == g)[0], diffs[k : k + 1])[
                            :, 0
                        ]
                measured[f"xfer_{field}"] = arr
            for key, value in measured.items():
                result[f"{fmt}_{key}"] = value
            del concept, antagonist, delta
        np.savez_compressed(
            dst,
            **result,
            auc_grid=np.array(draws["auc_grid"], np.int32),
            conv_grid=np.array(draws["conv_grid"], np.int32),
            n_rows=np.int32(n),
            formats=np.array(args.formats),
            lang_combos=np.array([[g in c for g in range(len(LANGUAGES))] for c in LANG_COMBOS]),
        )
        if pi % 10 == 0:
            a = result.get("assistant_auroc", next(iter(result.values())))
            log.info(
                "  %s/%s pair %s: %s rows; assistant AUROC(N=8) max-layer %.3f",
                pi,
                len(pairs),
                pid,
                n,
                float(np.nanmax(a[:, draws["auc_grid"].index(8)])),
            )
    log.info("=== shard %s done ===", args.shard)


if __name__ == "__main__":
    main()
