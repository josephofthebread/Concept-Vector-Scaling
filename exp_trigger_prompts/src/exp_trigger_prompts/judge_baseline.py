from __future__ import annotations

import collections
import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from .judge import sampled_slots, score_metric

log = logging.getLogger("trigger-judge-baseline")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = ArgumentParser()
    p.add_argument("--sweep", type=Path, required=True, help="merged steering sweep jsonl")
    p.add_argument("--features", type=Path, default=Path("data/features.json"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--prompts", type=int, default=8)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--reasoning", default="low")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--limit", type=int, default=0, help="cap rows for a throughput probe (0 = all)")
    args = p.parse_args()

    onto = {f["pair_id"]: f for f in json.loads(args.features.read_text())["features"]}
    by_pair_slots: dict[int, set[int]] = collections.defaultdict(set)
    rows: list[dict[str, Any]] = []
    for line in args.sweep.open():
        r = json.loads(line)
        if r["alpha"] != 0.0:
            continue
        by_pair_slots[r["pair_id"]].add(r["slot"])
        rows.append(r)
    keep_slots = {p: sampled_slots(list(s), args.prompts) for p, s in by_pair_slots.items()}
    keep_pairs = set(sorted(by_pair_slots)[args.shard :: args.shards])
    chats: list[dict[str, Any]] = []
    for r in rows:
        pid = r["pair_id"]
        if pid not in keep_pairs or r["slot"] not in keep_slots[pid]:
            continue
        f = onto[pid]
        chats.append(
            {
                "key": f"{pid}:{r['slot']}",
                "pair_id": pid,
                "slot": r["slot"],
                "concept": f["concept"],
                "antagonist": f["antagonist"],
                "guidance": f["narrative_guidance"],
                "prompt": r["prompt"],
                "response": r["response"],
            }
        )
    chats.sort(key=lambda c: (c["pair_id"], c["slot"]))
    if args.limit:
        chats = chats[: args.limit]
    log.info(
        "shard %d/%d: %d concepts, %d baseline rows (x4 passes)",
        args.shard,
        args.shards,
        len({c["pair_id"] for c in chats}),
        len(chats),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cpos = [{**c, "pole": c["concept"]} for c in chats]
    cneg = [{**c, "pole": c["antagonist"]} for c in chats]

    from vllm import LLM, SamplingParams

    engine = LLM(
        model="openai/gpt-oss-120b",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    tkw = {"reasoning_effort": args.reasoning}

    s_cpos = score_metric(
        engine, sampling, tkw, "concept", cpos, args.out_dir / f"concept-pos-shard-{args.shard}.jsonl"
    )
    s_cneg = score_metric(
        engine, sampling, tkw, "concept", cneg, args.out_dir / f"concept-neg-shard-{args.shard}.jsonl"
    )
    s_ins = score_metric(engine, sampling, tkw, "instruct", chats, args.out_dir / f"instruct-shard-{args.shard}.jsonl")
    s_flu = score_metric(engine, sampling, tkw, "fluency", chats, args.out_dir / f"fluency-shard-{args.shard}.jsonl")

    merged = args.out_dir / f"baseline-shard-{args.shard}.jsonl"
    with merged.open("w") as sink:
        for c in chats:
            k = c["key"]
            sink.write(
                json.dumps(
                    {
                        "pair_id": c["pair_id"],
                        "slot": c["slot"],
                        "concept_pos": s_cpos.get(k, -1),
                        "concept_neg": s_cneg.get(k, -1),
                        "instruct": s_ins.get(k, -1),
                        "fluency": s_flu.get(k, -1),
                    }
                )
                + "\n"
            )
    log.info("wrote %s (%d rows)", merged, len(chats))


if __name__ == "__main__":
    main()
