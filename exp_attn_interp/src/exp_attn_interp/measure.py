from __future__ import annotations

import json
import logging
import time
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file

from exp_attn_interp.common import (
    ValueCatcher,
    build_positions,
    center,
    generate_reply,
    load_model,
    sink_mask,
    token_cosine,
)
from exp_attn_interp.config import (
    GRID_POSITIONS,
    LMSYS_ITEMS,
    MARKER_MIN_OCCURRENCES,
    MARKERS_PRESENCE,
    MIN_CONTEXT,
    NUM_BLOCKS,
    PAIRS,
    READ_LAYER,
    REPLY_TOKENS,
    SAMPLING,
    generation_seed,
)
from exp_attn_interp.stats import rank_auroc, spearman_matrix
from exp_attn_interp.vectors import load_diff_vectors

log = logging.getLogger("measure")


def marker_bridge(
    weighted_by_block: list[torch.Tensor], context_ids: list[int], marker_ids: list[set[int]], heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """AUROC[36, 32, pairs] of head attention on concept-k marker tokens vs the rest (NaN if absent),
    and count[pairs] of marker tokens present. `weighted_by_block[b]` is [heads, |ctx|]."""
    stacked = torch.stack(weighted_by_block, dim=0).reshape(NUM_BLOCKS * heads, -1)
    auc = torch.full((NUM_BLOCKS * heads, PAIRS), float("nan"), device=stacked.device)
    counts = torch.zeros(PAIRS, dtype=torch.int32)
    for k in range(PAIRS):
        positive = torch.tensor([t in marker_ids[k] for t in context_ids], device=stacked.device)
        n = int(positive.sum().item())
        counts[k] = n
        if MARKER_MIN_OCCURRENCES <= n < len(context_ids):
            auc[:, k] = rank_auroc(stacked, positive)
    return auc.reshape(NUM_BLOCKS, heads, PAIRS), counts


def process_item(model, tokenizer, catcher, vectors, item, out_dir, marker_ids, special_ids):
    rho_path = out_dir / "measure" / f"item-{item['item_id']:04d}.safetensors"
    if rho_path.exists():
        return {"item": item["item_id"], "skipped": True}
    seed = generation_seed(int(item["item_id"]))
    prompt_ids, reply_ids = generate_reply(
        model, tokenizer, item["text"], int(item["item_id"]), REPLY_TOKENS, SAMPLING, seed
    )
    ids = prompt_ids + reply_ids
    positions = build_positions(tokenizer, ids, len(prompt_ids), GRID_POSITIONS)

    with torch.inference_mode():
        out = model(
            input_ids=torch.tensor([ids], device=model.device),
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden = list(out.hidden_states)
    mask = sink_mask(list(out.attentions), hidden)

    heads = catcher.heads
    rho = torch.zeros(len(positions), NUM_BLOCKS, heads, PAIRS, dtype=torch.float16)
    rho_deshared = torch.zeros(len(positions), NUM_BLOCKS, heads, PAIRS, dtype=torch.float16)
    rho_raw_assist = torch.zeros(NUM_BLOCKS, heads, PAIRS, dtype=torch.float16)
    assist_index = next((i for i, (_, kind) in enumerate(positions) if kind == "role_assistant"), None)
    assist_weighted: list[torch.Tensor] = []
    assist_deshared: list[torch.Tensor] = []
    assist_context: list[int] = []

    for block in range(NUM_BLOCKS):
        attn = out.attentions[block][0].to(torch.float32)
        norms = catcher.value_norms(block)
        cos = token_cosine(hidden[block + 1][0], vectors[block + 1])
        z = center(cos, "double")
        for index, (position, _kind) in enumerate(positions):
            context = torch.arange(position, device=attn.device)[~mask[:position]]
            if context.numel() < MIN_CONTEXT:
                continue
            weighted = attn[:, position, context] * norms[:, context]
            deshared = weighted - weighted.mean(dim=0, keepdim=True)
            rho[index, block] = spearman_matrix(weighted, z[context]).half().cpu()
            rho_deshared[index, block] = spearman_matrix(deshared, z[context]).half().cpu()
            if assist_index is not None and index == assist_index:
                rho_raw_assist[block] = spearman_matrix(weighted, cos[context]).half().cpu()
                assist_weighted.append(weighted)
                assist_deshared.append(deshared)
                if not assist_context:
                    assist_context = context.tolist()
        del attn

    marker_auc = torch.full((NUM_BLOCKS, heads, PAIRS), float("nan"), dtype=torch.float16)
    marker_auc_deshared = torch.full((NUM_BLOCKS, heads, PAIRS), float("nan"), dtype=torch.float16)
    counts = torch.zeros(PAIRS, dtype=torch.int32)
    if assist_weighted:
        context_ids = [ids[i] for i in assist_context]
        auc, counts = marker_bridge(assist_weighted, context_ids, marker_ids, heads)
        marker_auc = auc.half().cpu()
        auc_d, _ = marker_bridge(assist_deshared, context_ids, marker_ids, heads)
        marker_auc_deshared = auc_d.half().cpu()

    with torch.inference_mode():
        received = out.attentions[READ_LAYER][0].mean(dim=0).mean(dim=0)
        vnorm = catcher.value_norms(READ_LAYER).mean(dim=0)
    sink = mask
    raw_share = float(received[sink].sum() / received.sum().clamp_min(1e-9))
    vw = received * vnorm
    vw_share = float(vw[sink].sum() / vw.sum().clamp_min(1e-9))

    (out_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    (out_dir / "transcripts" / f"item-{item['item_id']:04d}.json").write_text(
        json.dumps(
            {
                "item_id": item["item_id"],
                "conversation_id": item["conversation_id"],
                "class_id": item["class_id"],
                "text": item["text"],
                "reply": tokenizer.decode(reply_ids, skip_special_tokens=True),
                "prompt_length": len(prompt_ids),
                "tokens": len(ids),
                "sink_tokens": int(mask.sum().item()),
                "positions": [{"position": p, "kind": k} for p, k in positions],
            },
            ensure_ascii=False,
        )
    )
    rho_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "rho": rho,
            "rho_deshared": rho_deshared,
            "rho_raw_assistant": rho_raw_assist,
            "marker_auc": marker_auc,
            "marker_auc_deshared": marker_auc_deshared,
            "marker_count": counts,
        },
        str(rho_path),
        metadata={
            "item": str(item["item_id"]),
            "construction": "diff",
            "tag": "assistant",
            "centering": "double",
            "positions": json.dumps([{"position": p, "kind": k} for p, k in positions]),
        },
    )
    return {
        "item": item["item_id"],
        "tokens": len(ids),
        "positions": len(positions),
        "sinks": int(mask.sum().item()),
        "marker_concepts": int((counts >= MARKER_MIN_OCCURRENCES).sum()),
        "sink_raw_share": round(raw_share, 3),
        "sink_vw_share": round(vw_share, 3),
        "bytes": rho_path.stat().st_size,
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--items", type=Path, default=LMSYS_ITEMS)
    parser.add_argument("--markers", type=Path, default=MARKERS_PRESENCE)
    parser.add_argument("--chunk", type=int, default=0, help="item_id // 64")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("work"))
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model, tokenizer = load_model(offline=args.offline)
    catcher = ValueCatcher(model.model)
    vectors = load_diff_vectors(device=str(model.device))
    markers = json.loads(args.markers.read_text())["concept"]
    marker_ids = [set(markers[str(k)]) for k in range(PAIRS)]
    special_ids = set(tokenizer.all_special_ids)

    items = [it for it in pq.read_table(args.items).to_pylist() if it["chunk"] == args.chunk]
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"no items in chunk {args.chunk}")
    started = time.monotonic()
    total = 0
    sink_raw: list[float] = []
    sink_vw: list[float] = []
    for number, item in enumerate(sorted(items, key=lambda i: i["item_id"]), start=1):
        result = process_item(model, tokenizer, catcher, vectors, item, args.out_dir, marker_ids, special_ids)
        total += result.get("bytes", 0)
        if "sink_raw_share" in result:
            sink_raw.append(result["sink_raw_share"])
            sink_vw.append(result["sink_vw_share"])
        log.info("%s/%s %s", number, len(items), result)
    if sink_raw:
        import statistics

        raw_m, vw_m = statistics.median(sink_raw), statistics.median(sink_vw)
        log.info(
            "sink diagnostic @L%d (median over items): raw share=%.3f, value-weighted share=%.3f, suppression=%.1fx",
            READ_LAYER,
            raw_m,
            vw_m,
            raw_m / max(vw_m, 1e-6),
        )
    log.info("chunk %s done: %.1f min, %.2f GiB", args.chunk, (time.monotonic() - started) / 60, total / 2**30)


if __name__ == "__main__":
    main()
