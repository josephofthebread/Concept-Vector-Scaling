from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from . import config

log = logging.getLogger("trigger-baselines")

CARRY = ("pair_id", "class_name", "kind", "pole_text", "concept", "antagonist", "slot", "seed")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, help="instruct checkpoint id")
    parser.add_argument("--prompts", type=Path, required=True, help="merged Stage-1 prompts.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="JSONL destination for this shard")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=config.BASELINE_MAX_TOKENS)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--limit", type=int, default=0, help="cap prompts for a partial run (0 = all)")
    args = parser.parse_args()

    records: list[dict] = []
    with args.prompts.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("ok") and record.get("prompt"):
                records.append(record)
    shard_records = records[args.shard :: args.shards]
    if args.limit:
        shard_records = shard_records[: args.limit]
    log.info(
        "%d usable prompts; shard %d/%d handles %d for %s",
        len(records),
        args.shard,
        args.shards,
        len(shard_records),
        args.model,
    )

    from vllm import LLM, SamplingParams

    engine = LLM(
        model=args.model,
        max_model_len=config.BASELINE_MAX_MODEL_LEN,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )

    conversations: list[Any] = [[{"role": "user", "content": record["prompt"]}] for record in shard_records]
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    log.info("generating %d greedy baselines", len(conversations))
    replies = engine.chat(
        conversations,
        sampling,
        chat_template_kwargs=config.chat_template_kwargs(args.model),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for record, reply in zip(shard_records, replies):
            row = {key: record[key] for key in CARRY}
            row["model"] = args.model
            row["prompt"] = record["prompt"]
            row["response"] = reply.outputs[0].text
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    log.info("wrote %d baselines to %s", len(shard_records), args.out)


if __name__ == "__main__":
    main()
