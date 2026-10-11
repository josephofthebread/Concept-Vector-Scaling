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
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp_suspicious_tests.chat import find_subsequence
from exp_suspicious_tests.config import (
    CHECKPOINTS,
    EXP4_GRID_POSITIONS,
    EXP4_MIN_CONTEXT,
    EXP4_REPLY_TOKENS,
    FAMILY_MARKERS,
    INSTRUCT_CHECKPOINTS,
    INSTRUCT_MODEL,
    PAIRS,
    SAMPLING,
    SINK_COSINE,
    SINK_EPSILON,
    SINK_LEADING_TOKENS,
    generation_seed,
    model_family,
)
from exp_suspicious_tests.stats import spearman_matrix
from exp_suspicious_tests.vectors import load_vectors_for

log = logging.getLogger("exp4")


class ValueCatcher:
    """Hooks every layer's v_proj and turns values into per-query-head output norms."""

    def __init__(self, model: Any) -> None:
        config = model.config
        self.heads = int(config.num_attention_heads)
        self.kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // self.heads))
        self.values: list[torch.Tensor | None] = [None] * len(model.layers)
        self.grams: list[torch.Tensor] = []
        for index, block in enumerate(model.layers):
            block.self_attn.v_proj.register_forward_hook(self._catch(index))
            weight = block.self_attn.o_proj.weight
            slices = weight.view(weight.shape[0], self.heads, self.head_dim).to(torch.float32)
            self.grams.append(torch.einsum("ohd,ohe->hde", slices, slices))

    def _catch(self, index: int) -> Any:
        def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            self.values[index] = output.detach()

        return hook

    def value_norms(self, index: int) -> torch.Tensor:
        """[heads, T] — ||W_O_head @ v_kv(head)(t)|| via the per-head Gram matrix."""
        raw = self.values[index]
        if raw is None:
            raise RuntimeError(f"v_proj hook missed layer {index}")
        tokens = raw.shape[1]
        values = raw.view(tokens, self.kv_heads, self.head_dim).to(torch.float32)
        group = self.heads // self.kv_heads
        gram = self.grams[index].to(raw.device)
        norms = torch.empty(self.heads, tokens, device=raw.device, dtype=torch.float32)
        for head in range(self.heads):
            v = values[:, head // group]
            quad = torch.einsum("td,de,te->t", v, gram[head], v)
            norms[head] = quad.clamp_min(0).sqrt()
        return norms


def process_item(
    model: Any,
    tokenizer: Any,
    catcher: ValueCatcher,
    vectors: torch.Tensor,
    item: dict[str, Any],
    out_dir: Path,
    markers: dict[str, str],
) -> dict[str, Any]:
    rho_path = out_dir / "rho" / f"item-{item['item_id']:04d}.safetensors"
    transcript_path = out_dir / "transcripts" / f"item-{item['item_id']:04d}.json"
    if rho_path.exists():
        return {"item": item["item_id"], "skipped": True}

    messages = [{"role": "user", "content": item["text"]}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    torch.manual_seed(generation_seed(int(item["item_id"])))
    batch = torch.tensor([prompt_ids], device=model.device)
    with torch.inference_mode():
        generated = model.generate(
            batch,
            attention_mask=torch.ones_like(batch),
            max_new_tokens=EXP4_REPLY_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            **SAMPLING,
        )
    reply_ids = generated[0, len(prompt_ids) :].tolist()
    ids = prompt_ids + reply_ids
    total = len(ids)
    im_start = tokenizer(markers["turn_start"], add_special_tokens=False)["input_ids"][0]
    im_end = tokenizer(markers["turn_end"], add_special_tokens=False)["input_ids"][0]
    user_ids = tokenizer(item["text"], add_special_tokens=False)["input_ids"]
    user_offset = find_subsequence(ids, user_ids)
    if user_offset < 0 and len(user_ids) > 1:
        user_ids = user_ids[1:]
        user_offset = find_subsequence(ids, user_ids)

    special: list[tuple[int, str]] = []
    header_roles = [index + 1 for index, token in enumerate(ids[:-1]) if token == im_start]
    if header_roles:
        special.append((header_roles[0], "role_user"))
    if len(header_roles) > 1:
        special.append((header_roles[-1], "role_assistant"))
    if user_offset >= 0:
        special.append((user_offset + len(user_ids) - 1, "user_last"))
    special += [(index, "im_end") for index, token in enumerate(ids) if token == im_end]
    if len(prompt_ids) < total:
        special.append((len(prompt_ids), "reply_first"))
    special.append((total - 1, "last"))

    taken = {position for position, _ in special}
    grid = torch.linspace(EXP4_MIN_CONTEXT, total - 2, EXP4_GRID_POSITIONS).round().long().tolist()
    positions = [(position, kind) for position, kind in special if position >= EXP4_MIN_CONTEXT]
    positions += [(position, "grid") for position in sorted(set(grid)) if position not in taken]
    positions.sort(key=lambda pair: pair[0])

    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([ids], device=model.device),
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden = list(output.hidden_states)
    attentions = list(output.attentions)
    tokens = attentions[0].shape[-1]
    mask = torch.zeros(tokens, dtype=torch.bool, device=attentions[0].device)
    mask[:SINK_LEADING_TOKENS] = True
    for index, attn in enumerate(attentions):
        received = attn[0].mean(dim=0).mean(dim=0)
        candidates = (received > SINK_EPSILON).nonzero().flatten()
        state = hidden[index][0].to(torch.float32)
        primary = torch.nn.functional.normalize(state[0], dim=0)
        for token in candidates.tolist():
            if token < SINK_LEADING_TOKENS:
                continue
            if torch.nn.functional.normalize(state[token], dim=0) @ primary > SINK_COSINE:
                mask[token] = True

    blocks = len(output.attentions)
    rho = torch.zeros(len(positions), blocks, catcher.heads, PAIRS, dtype=torch.float16)
    kept = []
    for block in range(blocks):
        attn = output.attentions[block][0].to(torch.float32)
        norms = catcher.value_norms(block)
        act = hidden[block][0].to(torch.float32) @ vectors[block].T
        for index, (position, _kind) in enumerate(positions):
            context = torch.arange(position, device=attn.device)[~mask[:position]]
            if context.numel() < EXP4_MIN_CONTEXT:
                continue
            weighted = attn[:, position, context] * norms[:, context]
            rho[index, block] = spearman_matrix(weighted, act[context]).half().cpu()
            if block == 0:
                kept.append(index)
        del attn

    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps(
            {
                **{key: item[key] for key in ("item_id", "conversation_id", "class_id", "text")},
                "reply": tokenizer.decode(reply_ids, skip_special_tokens=True),
                "input_ids": ids,
                "prompt_length": len(prompt_ids),
                "positions": [{"position": position, "kind": kind} for position, kind in positions],
                "sink_tokens": int(mask.sum().item()),
            },
            ensure_ascii=False,
        )
    )
    rho_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"rho": rho},
        str(rho_path),
        metadata={
            "item": str(item["item_id"]),
            "positions": json.dumps([{"position": p, "kind": k} for p, k in positions]),
            "construction": "diff",
            "tag": "assistant",
        },
    )
    return {
        "item": item["item_id"],
        "tokens": len(ids),
        "positions": len(positions),
        "measured": len(kept),
        "sinks": int(mask.sum().item()),
        "bytes": rho_path.stat().st_size,
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL, choices=INSTRUCT_CHECKPOINTS)
    parser.add_argument("--presentation", default="assistant")
    parser.add_argument("--items", type=Path, default=Path("data/lmsys_items.parquet"))
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("work/exp4"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not CHECKPOINTS[args.model].instruct:
        raise SystemExit("experiment 4 transcripts need a chat model")
    markers = FAMILY_MARKERS[model_family(args.model)]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.offline, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=args.offline,
        device_map={"": "cuda"},
    ).eval()
    catcher = ValueCatcher(model.model)
    vectors = load_vectors_for(args.model, args.presentation)["diff"]

    items = [item for item in pq.read_table(args.items).to_pylist() if item["chunk"] == args.chunk]
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"no items in chunk {args.chunk}")
    started = time.monotonic()
    total_bytes = 0
    for number, item in enumerate(sorted(items, key=lambda i: i["item_id"]), start=1):
        result = process_item(model, tokenizer, catcher, vectors, item, args.out_dir, markers)
        total_bytes += result.get("bytes", 0)
        log.info("%s/%s %s", number, len(items), result)
    elapsed = time.monotonic() - started
    log.info(
        "chunk %s done: %.1f min, %.2f GiB written (%.1f s/item)",
        args.chunk,
        elapsed / 60,
        total_bytes / 2**30,
        elapsed / len(items),
    )


if __name__ == "__main__":
    main()
