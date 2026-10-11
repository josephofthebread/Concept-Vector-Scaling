from __future__ import annotations

import json
import logging
import time
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file

from exp_attn_interp.common import ValueCatcher, generate_reply, load_model, sink_mask
from exp_attn_interp.config import (
    INSTRUCT_MODEL,
    LMSYS_ITEMS,
    MARKER_MIN_OCCURRENCES,
    MARKERS_PRESENCE,
    PAIRS,
    REPLY_TOKENS,
    SAMPLING,
    generation_seed,
    vector_tag_for,
)
from exp_attn_interp.vectors import load_diff_vectors

log = logging.getLogger("write")


STORY_PROMPT = "Generate a short story."


def story_ids(tokenizer, story: str) -> list[int]:
    """Render a story exactly as the assistant-presentation vectors saw it: teacher-forced reply."""
    messages = [{"role": "user", "content": STORY_PROMPT}, {"role": "assistant", "content": story}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def process_item(model, tokenizer, catcher, vectors, item, out_dir, marker_ids, special_ids):
    n_blocks = len(model.model.layers)
    path = out_dir / "write" / f"item-{item['item_id']:04d}.safetensors"
    if path.exists():
        return {"item": item["item_id"], "skipped": True}
    if "story" in item:
        ids = story_ids(tokenizer, item["story"])
    else:
        seed = generation_seed(int(item["item_id"]))
        prompt_ids, reply_ids = generate_reply(
            model, tokenizer, item["text"], int(item["item_id"]), REPLY_TOKENS, SAMPLING, seed
        )
        ids = prompt_ids + reply_ids
    with torch.inference_mode():
        out = model(
            input_ids=torch.tensor([ids], device=model.device),
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden = list(out.hidden_states)
    device = model.device
    mask = sink_mask(list(out.attentions), hidden)
    content = ~mask
    for pos, tok in enumerate(ids):
        if tok in special_ids:
            content[pos] = False
    cidx = content.nonzero().flatten()
    if cidx.numel() < 16:
        return {"item": item["item_id"], "skipped": "too few content tokens"}

    heads, kv, hd = catcher.heads, catcher.kv_heads, catcher.head_dim
    group = heads // kv
    T = len(ids)
    marker_m = torch.zeros(PAIRS, T, device=device)
    for k in range(PAIRS):
        mk = marker_ids[k]
        if not mk:
            continue
        for pos in cidx.tolist():
            if ids[pos] in mk:
                marker_m[k, pos] = 1.0
    n_marker = marker_m.sum(dim=1)
    eligible = n_marker >= MARKER_MIN_OCCURRENCES
    content_f = content.to(torch.float32)
    nonmarker_m = content_f[None, :] * (1.0 - marker_m)
    n_nonmarker = nonmarker_m.sum(dim=1)

    write_cos = torch.zeros(n_blocks, heads, PAIRS, dtype=torch.float16)
    write_norm = torch.zeros(n_blocks, heads, dtype=torch.float16)
    marker_write = torch.full((n_blocks, heads, PAIRS), float("nan"), dtype=torch.float16)
    marker_attend = torch.full((n_blocks, heads, PAIRS), float("nan"), dtype=torch.float16)
    marker_c = marker_m[:, cidx]
    n_pos = marker_c.sum(dim=1)
    n_neg = cidx.numel() - n_pos
    for b in range(n_blocks):
        attn = out.attentions[b][0].to(torch.float32)
        vals = catcher.values[b].view(T, kv, hd).to(torch.float32)
        recv = catcher.value_norms(b) * attn.sum(dim=1)
        rk = recv[:, cidx].argsort(dim=1).argsort(dim=1).to(torch.float32) + 1.0
        pos_rank = rk @ marker_c.T
        auc = (pos_rank - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg).clamp_min(1)
        auc[:, ~eligible] = float("nan")
        marker_attend[b] = auc.half().cpu()
        V = vals[:, torch.arange(heads, device=device) // group]
        ctx = torch.bmm(attn, V.permute(1, 0, 2))
        W_O = model.model.layers[b].self_attn.o_proj.weight.to(torch.float32)
        W_O = W_O.view(W_O.shape[0], heads, hd)
        w = torch.einsum("htd,ohd->hto", ctx, W_O)
        blk = model.model.layers[b]
        if hasattr(blk, "pre_feedforward_layernorm"):
            w = w * (1.0 + blk.post_attention_layernorm.weight.to(torch.float32))[None, None, :]
        norm = w.norm(dim=-1)
        wn = w / norm.clamp_min(1e-6).unsqueeze(-1)
        cos = wn @ vectors[b + 1].T
        cos = cos - cos.mean(dim=-1, keepdim=True)
        write_cos[b] = ((cos * content_f[None, :, None]).sum(1) / content_f.sum()).half().cpu()
        write_norm[b] = ((norm * content_f[None, :]).sum(1) / content_f.sum()).half().cpu()
        at_m = torch.einsum("htk,kt->hk", cos, marker_m) / n_marker.clamp_min(1).unsqueeze(0)
        at_n = torch.einsum("htk,kt->hk", cos, nonmarker_m) / n_nonmarker.clamp_min(1).unsqueeze(0)
        mw = at_m - at_n
        mw[:, ~eligible] = float("nan")
        marker_write[b] = mw.half().cpu()
        del attn, ctx, w, wn, cos

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "write_cos": write_cos,
            "write_norm": write_norm,
            "marker_write": marker_write,
            "marker_attend": marker_attend,
            "marker_count": n_marker.to(torch.int32).cpu(),
        },
        str(path),
        metadata={"item": str(item["item_id"]), "construction": "diff", "tag": "assistant"},
    )
    return {
        "item": item["item_id"],
        "tokens": T,
        "content": int(cidx.numel()),
        "marker_concepts": int(eligible.sum()),
        "bytes": path.stat().st_size,
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--items", type=Path, default=LMSYS_ITEMS)
    parser.add_argument("--markers", type=Path, default=MARKERS_PRESENCE)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("work"))
    parser.add_argument(
        "--vectors-file",
        type=Path,
        default=None,
        help="alternative [layers, pairs, hidden] 'diff' safetensors (e.g. sibling-orthogonalized)",
    )
    parser.add_argument(
        "--stories",
        type=int,
        default=0,
        help="stories mode: N English stories per pole per pair (chunk = 64 pairs), no generation",
    )
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    catcher = ValueCatcher(model.model)
    if args.vectors_file is not None:
        from safetensors.torch import load_file

        vectors = torch.nn.functional.normalize(
            load_file(str(args.vectors_file))["diff"].to(device=model.device, dtype=torch.float32), dim=-1
        )
        log.info("using vectors from %s", args.vectors_file)
    else:
        vectors = load_diff_vectors(device=str(model.device), tag=vector_tag_for(args.model))
    markers = json.loads(args.markers.read_text())["concept"]
    marker_ids = [set(markers[str(k)]) for k in range(PAIRS)]
    special_ids = set(tokenizer.all_special_ids)

    items: list[dict[str, Any]] = []
    if args.stories:
        from exp_attn_interp.config import GENDATA_ROOT

        tbl = pq.read_table(
            GENDATA_ROOT / "data" / "semantic.parquet",
            columns=["pair_id", "concept_text", "antagonist_text"],
            filters=[("language_id", "==", 0)],
        )
        by_pair: dict[int, list[tuple[str, str]]] = {}
        for p, c, a in zip(
            tbl["pair_id"].to_pylist(), tbl["concept_text"].to_pylist(), tbl["antagonist_text"].to_pylist()
        ):
            by_pair.setdefault(int(p), []).append((c, a))
        chunk_pairs = sorted(by_pair) if args.chunk < 0 else sorted(by_pair)[args.chunk * 64 : (args.chunk + 1) * 64]
        items = []
        for p in chunk_pairs:
            for i, (c, a) in enumerate(by_pair[p][: args.stories]):
                items.append({"item_id": 100000 + p * 20 + i * 2, "pair_id": p, "side": "concept", "story": c})
                items.append({"item_id": 100000 + p * 20 + i * 2 + 1, "pair_id": p, "side": "antagonist", "story": a})
        log.info("stories mode: %d pairs x %d stories/pole = %d items", len(chunk_pairs), args.stories, len(items))
    else:
        items = [it for it in pq.read_table(args.items).to_pylist() if it["chunk"] == args.chunk]
    if args.limit:
        items = items[: args.limit]
    started = time.monotonic()
    for number, item in enumerate(sorted(items, key=lambda i: int(i["item_id"])), start=1):
        result = process_item(model, tokenizer, catcher, vectors, item, args.out_dir, marker_ids, special_ids)
        log.info("%s/%s %s", number, len(items), result)
    log.info("chunk %s done: %.1f min", args.chunk, (time.monotonic() - started) / 60)


if __name__ == "__main__":
    main()
