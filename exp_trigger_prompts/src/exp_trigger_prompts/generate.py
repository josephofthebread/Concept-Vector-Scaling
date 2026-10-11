from __future__ import annotations

import json
import logging
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from . import config
from .features import Item, build_items, load_features

log = logging.getLogger("trigger-generate")

PROMPT_RE = re.compile(r"<PROMPT>\s*(.*?)\s*</PROMPT>", re.DOTALL | re.IGNORECASE)


def render(item: Item) -> list[dict[str, str]]:
    if item.kind == "neutral":
        user = config.build_neutral_prompt(item.concept, item.antagonist, item.guidance, item.hint)
    else:
        user = config.build_embodiment_prompt(item.pole_text, item.opposite_text, item.guidance, item.hint)
    return [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="JSONL destination for this shard")
    parser.add_argument("--shard", type=int, default=0, help="this shard's index")
    parser.add_argument("--shards", type=int, default=1, help="total number of shards")
    parser.add_argument("--effort", default=config.DRAFT_EFFORT, help="gpt-oss reasoning effort")
    parser.add_argument("--max-tokens", type=int, default=config.DRAFT_MAX_TOKENS)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--limit", type=int, default=0, help="cap items for a partial run (0 = all)")
    args = parser.parse_args()

    features = load_features()
    items = build_items(features)
    shard_items = items[args.shard :: args.shards]
    if args.limit:
        shard_items = shard_items[: args.limit]
    log.info(
        "%d features -> %d items total; shard %d/%d handles %d",
        len(features),
        len(items),
        args.shard,
        args.shards,
        len(shard_items),
    )

    from vllm import LLM, SamplingParams

    engine = LLM(
        model=config.DRAFT_MODEL,
        max_model_len=config.DRAFT_MAX_MODEL_LEN,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )

    prompts: list[Any] = [render(item) for item in shard_items]
    sampling = [
        SamplingParams(
            temperature=config.DRAFT_TEMPERATURE,
            top_p=config.DRAFT_TOP_P,
            top_k=config.DRAFT_TOP_K,
            seed=item.seed,
            max_tokens=args.max_tokens,
        )
        for item in shard_items
    ]

    log.info("drafting %d prompts (effort=%s)", len(prompts), args.effort)
    replies = engine.chat(prompts, sampling, chat_template_kwargs={"reasoning_effort": args.effort})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = failed = 0
    with args.out.open("w") as handle:
        for item, reply in zip(shard_items, replies):
            text = reply.outputs[0].text
            prompt = None
            for candidate in reversed(PROMPT_RE.findall(text)):
                cleaned = candidate.strip()
                if cleaned:
                    prompt = cleaned
                    break
            ok = prompt is not None
            failed += not ok
            record = {
                "pair_id": item.pair_id,
                "class_name": item.class_name,
                "kind": item.kind,
                "pole_text": item.pole_text,
                "concept": item.concept,
                "antagonist": item.antagonist,
                "slot": item.slot,
                "seed": item.seed,
                "hint": item.hint,
                "draft_model": config.DRAFT_MODEL,
                "effort": args.effort,
                "ok": ok,
                "prompt": prompt if ok else "",
            }
            if not ok:
                record["raw"] = text
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    log.info("wrote %d records to %s (%d failed to parse / refused)", written, args.out, failed)


if __name__ == "__main__":
    main()
