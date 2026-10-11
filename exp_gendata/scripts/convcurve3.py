from __future__ import annotations

import gc
import logging
import os
from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, DynamicCache

from exp_gendata.batching import mean_pool, pooling_selector
from exp_gendata.config import GENRES, LANGUAGES, SKIP_TOKENS, STORY_MAX_TOKENS
from exp_gendata.presentation import encode_wrapping

log = logging.getLogger("convcurve3")

BEST_LAYER = {"Qwen/Qwen3-8B": 22, "unsloth/Meta-Llama-3.1-8B-Instruct": 14, "google/gemma-2-9b-it": 22}
POWERS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
TEST = 25
REPEATS = 8
MATCHED_SIZE = 200
LANG_REPEATS = 4
LANG_COMBOS = [c for k in range(1, len(LANGUAGES)) for c in combinations(range(len(LANGUAGES)), k)]


def auroc_cols(scores: np.ndarray, n_pos: int) -> np.ndarray:
    """scores [story, subset], first n_pos rows positive -> AUROC per subset."""
    ranks = scores.argsort(axis=0).argsort(axis=0) + 1.0
    n_neg = scores.shape[0] - n_pos
    return (ranks[:n_pos].sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def weights(index_lists: list[np.ndarray], n: int) -> np.ndarray:
    w = np.zeros((len(index_lists), n), dtype=np.float32)
    for i, idx in enumerate(index_lists):
        w[i, idx] = 1.0 / len(idx)
    return w


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("dh,dh->d", a, b) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30)


def statistics(
    c: np.ndarray,
    a: np.ndarray,
    groups: dict,
    n_generators: int,
    rng: np.random.Generator,
    lang_combos: list = LANG_COMBOS,
) -> dict:
    """c, a [row, hidden] fp32 (row-aligned: matched stories). All subset vectors via GEMMs."""
    n = len(c)
    out: dict[str, np.ndarray] = {}
    full = c.mean(0) - a.mean(0)

    pc, pa = rng.permutation(n), rng.permutation(n)
    tc, ta, rc, ra = pc[:TEST], pa[:TEST], pc[TEST:], pa[TEST:]
    auc_grid = [N for N in POWERS if N < len(rc)] + [len(rc)]
    conv_grid = [N for N in POWERS if 2 * N <= n] + [n // 2]
    ci = [rng.choice(rc, N, replace=False) for N in auc_grid for _ in range(REPEATS)]
    ai = [rng.choice(ra, N, replace=False) for N in auc_grid for _ in range(REPEATS)]
    diffs = weights(ci, n) @ c - weights(ai, n) @ a
    test = np.concatenate([c[tc], a[ta]])
    out["auroc_u"] = auroc_cols(test @ diffs.T, TEST).reshape(len(auc_grid), REPEATS).mean(1)
    out["nested_u"] = cosine_rows(diffs, np.broadcast_to(full, diffs.shape)).reshape(len(auc_grid), REPEATS).mean(1)
    first_c, second_c, first_a, second_a = [], [], [], []
    for N in conv_grid:
        for _ in range(REPEATS):
            sc, sa = rng.permutation(n)[: 2 * N], rng.permutation(n)[: 2 * N]
            first_c.append(sc[:N])
            second_c.append(sc[N:])
            first_a.append(sa[:N])
            second_a.append(sa[N:])
    va = weights(first_c, n) @ c - weights(first_a, n) @ a
    vb = weights(second_c, n) @ c - weights(second_a, n) @ a
    out["conv_u"] = cosine_rows(va, vb).reshape(len(conv_grid), REPEATS).mean(1)

    delta = c - a
    order = rng.permutation(n)
    t, r = order[:TEST], order[TEST:]
    idx = [rng.choice(r, N, replace=False) for N in auc_grid for _ in range(REPEATS)]
    diffs = weights(idx, n) @ delta
    test = np.concatenate([c[t], a[t]])
    out["auroc_m"] = auroc_cols(test @ diffs.T, TEST).reshape(len(auc_grid), REPEATS).mean(1)
    firsts, seconds = [], []
    for N in conv_grid:
        for _ in range(REPEATS):
            pick = rng.permutation(n)[: 2 * N]
            firsts.append(pick[:N])
            seconds.append(pick[N:])
    out["conv_m"] = (
        cosine_rows(weights(firsts, n) @ delta, weights(seconds, n) @ delta).reshape(len(conv_grid), REPEATS).mean(1)
    )

    by_lang = {g: np.where(groups["language"] == g)[0] for g in range(len(LANGUAGES))}
    lists = [np.concatenate([by_lang[g] for g in combo]) for combo in lang_combos]
    for combo in lang_combos:
        share = np.array_split(np.arange(MATCHED_SIZE), len(combo))
        for _ in range(LANG_REPEATS):
            lists.append(
                np.concatenate(
                    [rng.choice(by_lang[g], min(len(s), len(by_lang[g])), replace=False) for g, s in zip(combo, share)]
                )
            )
    diffs = weights(lists, n) @ delta
    lang_all = np.full((len(lang_combos), len(LANGUAGES)), np.nan, dtype=np.float32)
    lang_size = np.full_like(lang_all, np.nan)
    for g in range(len(LANGUAGES)):
        rows = by_lang[g]
        if len(rows) < 8:
            continue
        auc = auroc_cols(np.concatenate([c[rows], a[rows]]) @ diffs.T, len(rows))
        for k, combo in enumerate(lang_combos):
            if g not in combo:
                lang_all[k, g] = auc[k]
                start = len(lang_combos) + k * LANG_REPEATS
                lang_size[k, g] = auc[start : start + LANG_REPEATS].mean()
    out["lang_all"], out["lang_size"] = lang_all, lang_size

    for field, size in (("genre", len(GENRES)), ("generator", n_generators)):
        arr = np.full(size, np.nan, dtype=np.float32)
        for g in range(size):
            held, train = np.where(groups[field] == g)[0], np.where(groups[field] != g)[0]
            if len(held) >= 8 and len(train) >= 8:
                diff = weights([train], n) @ delta
                arr[g] = auroc_cols(np.concatenate([c[held], a[held]]) @ diff.T, len(held))[0]
        out[f"xfer_{field}"] = arr

    for field, size in (("language", len(LANGUAGES)), ("genre", len(GENRES)), ("generator", n_generators)):
        same = np.full(size, np.nan, dtype=np.float32)
        other = np.full(size, np.nan, dtype=np.float32)
        count = np.zeros(size, dtype=np.int32)
        dircos = np.full(size, np.nan, dtype=np.float32)
        for g in range(size):
            inside, outside = np.where(groups[field] == g)[0], np.where(groups[field] != g)[0]
            if len(inside) < 8 or len(outside) < 8:
                continue
            v = weights([inside, outside], n) @ delta
            dircos[g] = cosine_rows(v[1:2], v[0:1])[0]
            test = rng.choice(inside, max(2, round(0.25 * len(inside))), replace=False)
            rest = np.setdiff1d(inside, test)
            N = min(len(rest), len(outside))
            lists = [rng.choice(rest, N, replace=False) for _ in range(LANG_REPEATS)]
            lists += [rng.choice(outside, N, replace=False) for _ in range(LANG_REPEATS)]
            auc = auroc_cols(np.concatenate([c[test], a[test]]) @ (weights(lists, n) @ delta).T, len(test))
            same[g], other[g] = auc[:LANG_REPEATS].mean(), auc[LANG_REPEATS:].mean()
            count[g] = N
        out[f"same_{field}"], out[f"other_{field}"] = same, other
        out[f"n_{field}"], out[f"dircos_{field}"] = count, dircos

    out["auc_grid"] = np.array(auc_grid, np.int32)
    out["conv_grid"] = np.array(conv_grid, np.int32)
    return {k: (v.astype(np.float32) if v.dtype.kind == "f" else v) for k, v in out.items()}


def _t_auroc(scores: torch.Tensor, n_pos: int) -> torch.Tensor:
    """scores [layer, story, subset], first n_pos stories positive -> AUROC [layer, subset]."""
    ranks = scores.argsort(dim=1).argsort(dim=1).double() + 1.0
    n_neg = scores.shape[1] - n_pos
    return (ranks[:, :n_pos].sum(1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _t_cos(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x * y).sum(-1) / (x.norm(dim=-1) * y.norm(dim=-1)).clamp_min(1e-30)


def statistics_gpu(c: torch.Tensor, a: torch.Tensor, groups: dict, n_generators: int, rng: np.random.Generator) -> dict:
    n, L = c.shape[0], c.shape[1]
    dev = c.device

    def mean_of(lists, x):
        return torch.einsum("sn,nlh->lsh", torch.from_numpy(weights(lists, n)).to(dev), x)

    def score(test, diffs):
        return torch.einsum("tlh,lsh->lts", test, diffs)

    out: dict[str, torch.Tensor] = {}
    full = c.mean(0) - a.mean(0)

    pc, pa = rng.permutation(n), rng.permutation(n)
    tc, ta, rc, ra = pc[:TEST], pa[:TEST], pc[TEST:], pa[TEST:]
    auc_grid = [N for N in POWERS if N < len(rc)] + [len(rc)]
    conv_grid = [N for N in POWERS if 2 * N <= n] + [n // 2]
    ci = [rng.choice(rc, N, replace=False) for N in auc_grid for _ in range(REPEATS)]
    ai = [rng.choice(ra, N, replace=False) for N in auc_grid for _ in range(REPEATS)]
    diffs = mean_of(ci, c) - mean_of(ai, a)
    test = torch.cat([c[torch.as_tensor(tc, device=dev)], a[torch.as_tensor(ta, device=dev)]])
    out["auroc_u"] = _t_auroc(score(test, diffs), TEST).view(L, len(auc_grid), REPEATS).mean(-1)
    out["nested_u"] = _t_cos(diffs, full[:, None]).view(L, len(auc_grid), REPEATS).mean(-1)
    first_c, second_c, first_a, second_a = [], [], [], []
    for N in conv_grid:
        for _ in range(REPEATS):
            sc, sa = rng.permutation(n)[: 2 * N], rng.permutation(n)[: 2 * N]
            first_c.append(sc[:N])
            second_c.append(sc[N:])
            first_a.append(sa[:N])
            second_a.append(sa[N:])
    va = mean_of(first_c, c) - mean_of(first_a, a)
    vb = mean_of(second_c, c) - mean_of(second_a, a)
    out["conv_u"] = _t_cos(va, vb).view(L, len(conv_grid), REPEATS).mean(-1)

    delta = c - a
    order = rng.permutation(n)
    t, r = order[:TEST], order[TEST:]
    idx = [rng.choice(r, N, replace=False) for N in auc_grid for _ in range(REPEATS)]
    diffs = mean_of(idx, delta)
    tt = torch.as_tensor(t, device=dev)
    out["auroc_m"] = _t_auroc(score(torch.cat([c[tt], a[tt]]), diffs), TEST).view(L, len(auc_grid), REPEATS).mean(-1)
    firsts, seconds = [], []
    for N in conv_grid:
        for _ in range(REPEATS):
            pick = rng.permutation(n)[: 2 * N]
            firsts.append(pick[:N])
            seconds.append(pick[N:])
    out["conv_m"] = _t_cos(mean_of(firsts, delta), mean_of(seconds, delta)).view(L, len(conv_grid), REPEATS).mean(-1)

    by_lang = {g: np.where(groups["language"] == g)[0] for g in range(len(LANGUAGES))}
    lists = [np.concatenate([by_lang[g] for g in combo]) for combo in LANG_COMBOS]
    for combo in LANG_COMBOS:
        share = np.array_split(np.arange(MATCHED_SIZE), len(combo))
        for _ in range(LANG_REPEATS):
            lists.append(
                np.concatenate(
                    [rng.choice(by_lang[g], min(len(s), len(by_lang[g])), replace=False) for g, s in zip(combo, share)]
                )
            )
    diffs = mean_of(lists, delta)
    lang_all = torch.full((L, len(LANG_COMBOS), len(LANGUAGES)), float("nan"), device=dev, dtype=torch.float64)
    lang_size = lang_all.clone()
    for g in range(len(LANGUAGES)):
        rows = by_lang[g]
        if len(rows) < 8:
            continue
        rr = torch.as_tensor(rows, device=dev)
        auc = _t_auroc(score(torch.cat([c[rr], a[rr]]), diffs), len(rows))
        for k, combo in enumerate(LANG_COMBOS):
            if g not in combo:
                lang_all[:, k, g] = auc[:, k]
                start = len(LANG_COMBOS) + k * LANG_REPEATS
                lang_size[:, k, g] = auc[:, start : start + LANG_REPEATS].mean(-1)
    out["lang_all"], out["lang_size"] = lang_all, lang_size

    for field, size in (("genre", len(GENRES)), ("generator", n_generators)):
        arr = torch.full((L, size), float("nan"), device=dev, dtype=torch.float64)
        for g in range(size):
            held, train = np.where(groups[field] == g)[0], np.where(groups[field] != g)[0]
            if len(held) >= 8 and len(train) >= 8:
                hh = torch.as_tensor(held, device=dev)
                arr[:, g] = _t_auroc(score(torch.cat([c[hh], a[hh]]), mean_of([train], delta)), len(held))[:, 0]
        out[f"xfer_{field}"] = arr

    for field, size in (("language", len(LANGUAGES)), ("genre", len(GENRES)), ("generator", n_generators)):
        same = torch.full((L, size), float("nan"), device=dev, dtype=torch.float64)
        other, dircos = same.clone(), same.clone()
        count = np.zeros(size, dtype=np.int32)
        for g in range(size):
            inside, outside = np.where(groups[field] == g)[0], np.where(groups[field] != g)[0]
            if len(inside) < 8 or len(outside) < 8:
                continue
            v = mean_of([inside, outside], delta)
            dircos[:, g] = _t_cos(v[:, 1], v[:, 0])
            test_rows = rng.choice(inside, max(2, round(0.25 * len(inside))), replace=False)
            rest = np.setdiff1d(inside, test_rows)
            N = min(len(rest), len(outside))
            lists = [rng.choice(rest, N, replace=False) for _ in range(LANG_REPEATS)]
            lists += [rng.choice(outside, N, replace=False) for _ in range(LANG_REPEATS)]
            tr = torch.as_tensor(test_rows, device=dev)
            auc = _t_auroc(score(torch.cat([c[tr], a[tr]]), mean_of(lists, delta)), len(test_rows))
            same[:, g], other[:, g] = auc[:, :LANG_REPEATS].mean(-1), auc[:, LANG_REPEATS:].mean(-1)
            count[g] = N
        out[f"same_{field}"], out[f"other_{field}"], out[f"dircos_{field}"] = same, other, dircos
        out[f"n_{field}"] = torch.from_numpy(count)

    result = {
        k: (v.cpu().numpy().astype(np.float32) if v.is_floating_point() else v.cpu().numpy()) for k, v in out.items()
    }
    result["auc_grid"] = np.array(auc_grid, np.int32)
    result["conv_grid"] = np.array(conv_grid, np.int32)
    return result


class ReadOnlyPrefixCache(DynamicCache):
    """Holds one shared prompt prefix (expanded, not copied, to the batch) and never stores the batch's own
    keys/values: attention sees prefix + current tokens, and each layer's KV is freed as soon as it is used."""

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        layer = self.layers[layer_idx]
        return torch.cat([layer.keys, key_states], dim=-2), torch.cat([layer.values, value_states], dim=-2)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(BEST_LAYER))
    parser.add_argument("--dataset", type=Path, default=Path("data/semantic.parquet"))
    parser.add_argument("--ontology", type=Path, default=Path("data/semantic.ontology.json"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument(
        "--presentation",
        default="assistant",
        help="comma-separated formats, all done in one job: assistant, raw, user, paired_assistant",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="full depth, every statistic at every layer (embedding + all blocks), computed on the GPU",
    )
    parser.add_argument(
        "--deadline-min",
        type=float,
        default=0.0,
        help="stop starting new pairs this many minutes after the job started",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.6,
        help="share of free GPU memory one batch may use (headroom for fragmentation of multi-GB blocks)",
    )
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument(
        "--fp16-validate",
        action="store_true",
        help="allow gemma-2 in fp16 (V100) only after reproducing an existing bf16 result on this GPU",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="skip torch.compile of the MLPs and RMSNorms (fuses their memory-bound element-wise ops)",
    )
    parser.add_argument(
        "--attn",
        default="auto",
        help="auto = flash-attn2 (kernels-community build) when available, else sdpa (eager for gemma-2)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    shared = Path("work/tmp").resolve()
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(shared / "inductor-cache"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(shared / "triton-cache"))
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
    import json
    import time
    from concurrent.futures import ThreadPoolExecutor

    job_start = float(
        os.environ.get("SLURM_JOB_START_TIME", time.time() - (300 if "SLURM_JOB_ID" in os.environ else 0))
    )

    formats = [f.strip() for f in args.presentation.split(",") if f.strip()]
    for f in formats:
        assert f in ("assistant", "raw", "user", "paired_assistant"), f
    tag = args.model.split("/")[-1]
    layer = BEST_LAYER[args.model]

    def out_dir(fmt: str, all_layers: bool) -> Path:
        return Path(f"work/convcurve3{'L' if all_layers else ''}-{tag}" + ("" if fmt == "assistant" else f"-{fmt}"))

    outs = {f: out_dir(f, args.all_layers) for f in formats}
    for d in outs.values():
        d.mkdir(parents=True, exist_ok=True)

    meta = {int(p["pair_id"]): p for p in json.loads(args.ontology.read_text())["pairs"]}
    pairs = sorted(meta)[args.shard :: args.shards]
    if args.reverse:
        pairs = pairs[::-1]

    def have(f: str, pid: int) -> bool:
        name = f"pair-{pid:04d}.npz"
        return (outs[f] / name).exists() or (not args.all_layers and (out_dir(f, True) / name).exists())

    todo = {pid: [f for f in formats if not have(f, pid)] for pid in pairs}
    pairs = [pid for pid in pairs if todo[pid]]
    if args.all_layers:
        pairs.sort(key=lambda pid: all((out_dir(f, False) / f"pair-{pid:04d}.npz").exists() for f in todo[pid]))
    validate_pid = None
    if args.fp16_validate:
        done_pids = sorted(int(f.stem.split("-")[1]) for f in outs[formats[0]].glob("pair-*.npz"))
        if done_pids:
            validate_pid = done_pids[0]
            todo[validate_pid] = [formats[0]]
            pairs = [validate_pid] + [pid for pid in pairs if pid != validate_pid]
    log.info(
        "%s %s layer %s shard %s/%s%s: %s pairs to do",
        args.model,
        formats,
        layer,
        args.shard,
        args.shards,
        " (reverse)" if args.reverse else "",
        len(pairs),
    )
    if not pairs:
        return
    table = pq.read_table(
        args.dataset,
        columns=["pair_id", "language_id", "genre_id", "generator_id", "concept_text", "antagonist_text"],
        filters=[("pair_id", "in", pairs)],
    ).to_pydict()
    rows_by_pair: dict[int, list[int]] = {}
    for i, pid in enumerate(table["pair_id"]):
        rows_by_pair.setdefault(int(pid), []).append(i)
    n_generators = int(pq.read_table(args.dataset, columns=["generator_id"]).column(0).to_numpy().max()) + 1

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)

    def prepare(pid: int) -> dict:
        """CPU work for one pair (runs in a background thread while the GPU works on the previous pair)."""
        chosen = rows_by_pair[pid]
        n = len(chosen)
        encoded = tokenizer(
            [table["concept_text"][i] for i in chosen] + [table["antagonist_text"][i] for i in chosen],
            add_special_tokens=False,
            truncation=True,
            max_length=STORY_MAX_TOKENS,
        )["input_ids"]
        pm = meta[pid]
        poles = (str(pm["concept"]), str(pm["antagonist"]))
        language = [int(table["language_id"][i]) for i in chosen]
        genre = [int(table["genre_id"][i]) for i in chosen]
        wrap = {}
        for f in todo[pid]:
            cache = {}
            for side in (0, 1):
                for g in set(zip(language, genre)):
                    cache[(side, *g)] = encode_wrapping(
                        tokenizer, f, poles[side], LANGUAGES[g[0]], GENRES[g[1]], pm
                    ).prefix
            wrap[f] = cache
        return {
            "pid": pid,
            "n": n,
            "stories": (encoded[:n], encoded[n:]),
            "language": language,
            "genre": genre,
            "wrap": wrap,
            "groups": {
                f: np.array([int(table[f"{f}_id"][i]) for i in chosen]) for f in ("language", "genre", "generator")
            },
        }

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    reference = "eager" if config.model_type == "gemma2" else "sdpa"
    if dtype == torch.float16 and config.model_type == "gemma2":
        log.info("gemma-2 in fp16 (no bf16 on this GPU)")
    attention = args.attn
    if attention == "auto":
        attention = reference
        if torch.cuda.get_device_capability()[0] >= 8:
            try:
                import transformers.integrations.hub_kernels as hub_kernels
                from kernels import get_local_kernel

                hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
                repo = hub / "kernels--kernels-community--flash-attn2"
                commit = (repo / "refs" / "v1").read_text().strip()
                local_kernel = get_local_kernel(repo / "snapshots" / commit)
                hub_get = hub_kernels.get_kernel

                def get_kernel(repo_id, *a, **k):
                    return (
                        local_kernel
                        if repo_id.split("@")[0] == "kernels-community/flash-attn2"
                        else hub_get(repo_id, *a, **k)
                    )

                hub_kernels.get_kernel = get_kernel
                attention = "kernels-community/flash-attn2"
            except Exception as error:
                log.warning("local flash-attn2 kernel not loadable (%s); using %s", str(error)[:300], reference)
    try:
        model = AutoModel.from_pretrained(
            args.model, dtype=dtype, attn_implementation=attention, local_files_only=True, device_map={"": "cuda"}
        ).eval()
    except Exception as error:
        log.warning("attention %s unavailable (%s); using %s", attention, error, reference)
        attention = reference
        model = AutoModel.from_pretrained(
            args.model, dtype=dtype, attn_implementation=attention, local_files_only=True, device_map={"": "cuda"}
        ).eval()
    log.info("attention: %s (reference %s) on %s", attention, reference, torch.cuda.get_device_name())
    core = getattr(model, "model", model)
    if not args.all_layers:
        core.layers = torch.nn.ModuleList(list(core.layers)[:layer])
    dev = core.embed_tokens.weight.device
    n_cap = len(core.layers) + 1 if args.all_layers else 1
    captured: list[torch.Tensor | None] = [None] * n_cap
    pooling: list[torch.Tensor | None] = [None]

    def make_hook(slot: int):
        def hook(_m, _i, output) -> None:
            if pooling[0] is None:
                return
            state = output[0] if isinstance(output, tuple) else output
            captured[slot] = mean_pool(state, pooling[0])

        return hook

    if args.all_layers:
        core.embed_tokens.register_forward_hook(make_hook(0))
        for position, block in enumerate(core.layers):
            block.register_forward_hook(make_hook(position + 1))
    else:
        core.layers[-1].register_forward_hook(make_hook(0))
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    use_prefix = [not args.no_prefix_cache]

    compiled: list[tuple[torch.nn.Module, object]] = []
    if not args.no_compile:
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        for block in core.layers:
            for name in (
                "mlp",
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ):
                module = getattr(block, name, None)
                if isinstance(module, torch.nn.Module):
                    module.compile(dynamic=True)
                    compiled.append((module, module._compiled_call_impl))

    def set_compiled(on: bool) -> None:
        for module, impl in compiled:
            module._compiled_call_impl = impl if on else None

    def make_prefix(prefix: tuple[int, ...]):
        """KV of the shared prefix, computed once (batch 1)."""
        if not prefix or not use_prefix[0]:
            return None
        cache = DynamicCache(config=model.config)
        pooling[0] = None
        model(input_ids=torch.tensor([prefix], device=dev), past_key_values=cache, use_cache=True)
        cache.__class__ = ReadOnlyPrefixCache
        cache.base = [
            (i, lay.keys, lay.values)
            for i, lay in enumerate(cache.layers)
            if getattr(lay, "keys", None) is not None and lay.keys.numel()
        ]
        cache.prefix_len = len(prefix)
        return cache

    def run_batch(seqs: list[tuple[tuple[int, ...], tuple[int, ...]]], prefix_cache) -> torch.Tensor:
        """seqs: (tail, story) token tuples after the shared prefix. Returns pooled [slot, seq, hidden] fp32."""
        width = max(len(t) + len(x) for t, x in seqs)
        ids = np.full((len(seqs), width), pad, dtype=np.int64)
        mask = np.zeros((len(seqs), width), dtype=np.int64)
        weights_np = np.zeros((len(seqs), 1, width), dtype=np.float32)
        for k, (t, x) in enumerate(seqs):
            ids[k, : len(t)] = t
            ids[k, len(t) : len(t) + len(x)] = x
            mask[k, : len(t) + len(x)] = 1
            weights_np[k, 0, len(t) : len(t) + len(x)] = pooling_selector(len(x), SKIP_TOKENS)
        ids_t = torch.from_numpy(ids).pin_memory().to(dev, non_blocking=True)
        mask_t = torch.from_numpy(mask).pin_memory().to(dev, non_blocking=True)
        pooling[0] = torch.from_numpy(weights_np).pin_memory().to(dev, non_blocking=True).to(dtype)
        kwargs = {}
        if prefix_cache is not None:
            for i, k, v in prefix_cache.base:
                lay = prefix_cache.layers[i]
                lay.keys, lay.values = k.expand(len(seqs), -1, -1, -1), v.expand(len(seqs), -1, -1, -1)
            mask_t = torch.cat(
                [torch.ones(len(seqs), prefix_cache.prefix_len, dtype=mask_t.dtype, device=dev), mask_t], 1
            )
            kwargs = {"past_key_values": prefix_cache, "use_cache": True}
        else:
            kwargs = {"use_cache": False}
        try:
            model(input_ids=ids_t, attention_mask=mask_t, **kwargs)
        finally:
            pooling[0] = None
        return torch.stack(captured)

    budget = [16384]
    calibrated = [False]
    cal_width = [1]

    def forward_all(seqs, prefix_cache) -> torch.Tensor:
        """All sequences of one (pair, format), length-sorted, batched to the calibrated token budget."""
        lengths = np.array([len(t) + len(x) for t, x in seqs])
        order = np.argsort(lengths, kind="stable")
        result = torch.empty((n_cap, len(seqs), config.hidden_size), dtype=torch.float32, device=dev)

        def go(positions: list[int]) -> None:
            failure = None
            try:
                result[:, torch.as_tensor(positions, device=dev)] = run_batch(
                    [seqs[i] for i in positions], prefix_cache
                )
                return
            except torch.cuda.OutOfMemoryError:
                failure = "oom"
            except Exception as error:
                if not compiled:
                    raise
                failure = f"compile: {str(error)[:300]}"
            ##
            # NOTE: retries run outside the except block on purpose: a live exception keeps
            # the failed forward's activations (GBs) alive through its traceback.
            ##
            captured[:] = [None] * n_cap
            pooling[0] = None
            gc.collect()
            torch.cuda.empty_cache()
            if failure == "oom":
                if len(positions) == 1:
                    raise torch.cuda.OutOfMemoryError(
                        f"a single sequence ({int(lengths[positions[0]])} tokens) does not fit"
                    )
                budget[0] = int(budget[0] * 0.8)
                log.info("OOM at %s seqs -> split; budget now %s tokens", len(positions), budget[0])
                half = len(positions) // 2
                go(positions[:half])
                go(positions[half:])
            else:
                log.warning("compiled forward failed mid-run (%s): continuing uncompiled", failure)
                set_compiled(False)
                compiled.clear()
                go(positions)

        if not calibrated[0]:
            width = int(lengths[order[-1]])
            probe = [int(i) for i in order[-max(1, 16384 // width) :]]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            go(probe)
            torch.cuda.synchronize()
            per_token = (torch.cuda.max_memory_allocated() - base) / (len(probe) * width)
            free = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            budget[0] = max(2048, int(args.memory_fraction * free / per_token))
            cal_width[0] = width
            calibrated[0] = True
            log.info(
                "calibrated on %s: %.1f KB/padded token, %.1f GB free -> token budget %s",
                torch.cuda.get_device_name(),
                per_token / 1024,
                free / 2**30,
                budget[0],
            )

        def limit(width: int) -> float:
            if not attention.startswith("kernels-community/") and width > cal_width[0]:
                return budget[0] * cal_width[0] / width
            return budget[0]

        batch, widest = [], 0
        for i in order.tolist():
            w = max(widest, int(lengths[i]))
            if batch and (len(batch) + 1) * w > limit(w):
                go(batch)
                batch, w = [], int(lengths[i])
            batch.append(i)
            widest = w
        if batch:
            go(batch)
        return result

    done, checked_stats, checked_forward = 0, False, False
    pool = ThreadPoolExecutor(max_workers=1)
    upcoming = pool.submit(prepare, pairs[0])
    for index, pid in enumerate(pairs):
        if args.deadline_min and (time.time() - job_start) / 60 > args.deadline_min:
            log.info("deadline reached after %s pairs", done)
            break
        item = upcoming.result()
        if index + 1 < len(pairs):
            upcoming = pool.submit(prepare, pairs[index + 1])
        n, groups = item["n"], item["groups"]
        for fmt in todo[pid]:
            dst = outs[fmt] / f"pair-{pid:04d}.npz"
            if have(fmt, pid) and pid != validate_pid:
                continue
            t0 = time.time()
            full = {
                (side, k): item["wrap"][fmt][(side, item["language"][k], item["genre"][k])]
                for side in (0, 1)
                for k in range(n)
            }
            sequences = list(set(full.values()))
            first, shortest = sequences[0], min(len(x) for x in sequences)
            p = 0
            while p < shortest and all(x[p] == first[p] for x in sequences):
                p += 1
            seqs = [(full[(side, k)][p:], tuple(item["stories"][side][k])) for side in (0, 1) for k in range(n)]
            with torch.inference_mode():
                prefix_cache = make_prefix(next(iter(full.values()))[:p])
                if not checked_forward:
                    sample = list(range(0, len(seqs), max(1, len(seqs) // 8)))[:8]
                    try:
                        fast = run_batch([seqs[i] for i in sample], prefix_cache)
                    except Exception as error:
                        if not compiled:
                            raise
                        log.warning("compiled forward failed (%s): running uncompiled", str(error)[:300])
                        set_compiled(False)
                        compiled.clear()
                        fast = run_batch([seqs[i] for i in sample], prefix_cache)
                    set_compiled(False)
                    switched = False
                    if attention != reference:
                        try:
                            model.set_attn_implementation(reference)
                            switched = True
                        except Exception as error:
                            log.warning("cannot switch to %s for the self-check (%s)", reference, error)
                    plain = run_batch([(full[divmod(i, n)][:p] + seqs[i][0], seqs[i][1]) for i in sample], None)
                    if switched:
                        model.set_attn_implementation(attention)
                    set_compiled(True)
                    cos = torch.nn.functional.cosine_similarity(fast, plain, dim=-1).min().item()
                    rel = ((fast - plain).norm(dim=-1) / plain.norm(dim=-1).clamp_min(1e-12)).max().item()
                    if cos < 0.999 or rel > 0.05:
                        log.warning(
                            "fast-path self-check FAILED (min cos %.5f, max rel %.3f): falling back to %s, "
                            "no prefix cache, no compile",
                            cos,
                            rel,
                            reference,
                        )
                        set_compiled(False)
                        compiled.clear()
                        if attention != reference:
                            try:
                                model.set_attn_implementation(reference)
                                attention = reference
                            except Exception:
                                raise RuntimeError("fast path failed its self-check and the kernel cannot be switched")
                        use_prefix[0] = False
                        prefix_cache = None
                    else:
                        log.info(
                            "fast-path self-check ok (%s + %s-token prefix cache + %s compiled modules vs %s plain): "
                            "min cos %.6f, max rel diff %.4f over %s layers",
                            attention,
                            p if prefix_cache is not None else 0,
                            len(compiled),
                            reference,
                            cos,
                            rel,
                            n_cap,
                        )
                    checked_forward = True
                pooled = forward_all(seqs, prefix_cache)
                c = pooled[:, :n].permute(1, 0, 2).contiguous()
                a = pooled[:, n:].permute(1, 0, 2).contiguous()
                del pooled, prefix_cache
                t1 = time.time()
                stats = statistics_gpu(c, a, groups, n_generators, np.random.default_rng([args.seed, pid]))
                if not checked_stats:
                    slot = layer if args.all_layers else 0
                    ref = statistics(
                        c[:, slot].cpu().numpy(),
                        a[:, slot].cpu().numpy(),
                        groups,
                        n_generators,
                        np.random.default_rng([args.seed, pid]),
                    )
                    rank_keys = ("auroc_", "lang_", "xfer_", "same_", "other_")
                    worst, worst_rank = 0.0, 0.0
                    for key, value in ref.items():
                        mine = (
                            stats[key]
                            if (key in ("auc_grid", "conv_grid") or key.startswith("n_"))
                            else stats[key][slot]
                        )
                        mine, value = np.asarray(mine, float), np.asarray(value, float)
                        if not np.array_equal(np.isnan(value), np.isnan(mine)):
                            raise RuntimeError(f"stats self-check: NaN pattern differs in {key}")
                        if value.size:
                            diff = float(np.nanmax(np.abs(value - mine)))
                            if key.startswith(rank_keys):
                                worst_rank = max(worst_rank, diff)
                            else:
                                worst = max(worst, diff)
                    if worst > 2e-3 or worst_rank > 0.02:
                        raise RuntimeError(
                            f"stats self-check failed: cosine max diff {worst:.2e}, AUROC max diff {worst_rank:.2e}"
                        )
                    log.info(
                        "stats self-check vs numpy at layer %s: cosine max diff %.1e, AUROC max diff %.1e (ok)",
                        layer,
                        worst,
                        worst_rank,
                    )
                    checked_stats = True
                del c, a
            if args.all_layers:
                stats["layers"] = np.arange(n_cap, dtype=np.int32)
            else:
                stats = {
                    k: (v if (k in ("auc_grid", "conv_grid") or k.startswith("n_")) else v[0]) for k, v in stats.items()
                }
            t2 = time.time()
            if pid == validate_pid:
                ref = np.load(dst)
                rank_keys = ("auroc_", "lang_", "xfer_", "same_", "other_")
                worst_cos, worst_auc, bad = 0.0, 0.0, []
                for key in ref.files:
                    if key not in stats or ref[key].dtype.kind != "f":
                        continue
                    mine, theirs = np.asarray(stats[key], float), np.asarray(ref[key], float)
                    if np.isnan(mine).sum() > np.isnan(theirs).sum():
                        bad.append(f"{key}: NaN")
                        continue
                    diff = float(np.nanmax(np.abs(mine - theirs))) if theirs.size else 0.0
                    if key.startswith(rank_keys):
                        worst_auc = max(worst_auc, diff)
                    else:
                        worst_cos = max(worst_cos, diff)
                if bad or worst_cos > 5e-3 or worst_auc > 0.03:
                    raise SystemExit(
                        f"fp16 validation FAILED on pair {pid} {fmt}: cosine diff {worst_cos:.2e}, "
                        f"AUROC diff {worst_auc:.2e}, {bad[:3]}"
                    )
                log.info(
                    "fp16 validation vs stored bf16 pair %s %s: cosine max diff %.1e, AUROC max diff %.1e (ok)",
                    pid,
                    fmt,
                    worst_cos,
                    worst_auc,
                )
                continue
            tmp = outs[fmt] / f".pair-{pid:04d}.partial{os.getpid()}"
            with open(tmp, "wb") as handle:
                np.savez_compressed(
                    handle,
                    **stats,
                    layer=np.int32(layer),
                    n_rows=np.int32(n),
                    dtype=np.array(str(dtype).replace("torch.", "")),
                    gpu=np.array(torch.cuda.get_device_name()),
                )
            os.replace(tmp, dst)
            if True:
                tokens = sum(len(t) + len(x) for t, x in seqs)
                g8, c128 = list(stats["auc_grid"]).index(8), list(stats["conv_grid"]).index(128)
                pick = (lambda x: x[layer]) if args.all_layers else (lambda x: x)
                log.info(
                    "  pair %s %s: AUROC_u(8) %.3f conv_u(128) %.3f conv_m(500) %.4f | %s tok (+%s-tok prefix "
                    "cached) forward %.1fs (%.0f tok/s) stats %.1fs | %.1f min",
                    pid,
                    fmt,
                    pick(stats["auroc_u"])[g8],
                    pick(stats["conv_u"])[c128],
                    pick(stats["conv_m"])[-1],
                    tokens,
                    p if use_prefix[0] else 0,
                    t1 - t0,
                    tokens / max(t1 - t0, 1e-9),
                    t2 - t1,
                    (time.time() - job_start) / 60,
                )
            done += 1
    pool.shutdown(wait=False, cancel_futures=True)
    log.info("=== %s shard %s done: %s new (pair, format) results ===", tag, args.shard, done)


if __name__ == "__main__":
    main()
