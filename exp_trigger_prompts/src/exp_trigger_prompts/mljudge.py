from __future__ import annotations

import glob
import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from .judge import _ANSWER, CHUNK, CONCEPT_SYS, FLUENCY_SYS, concept_user, fluency_user, parse_rating

log = logging.getLogger("mljudge")
MODEL = "openai/gpt-oss-120b"

CODESWITCH_SYS = (
    "Please act as an impartial judge. The text fragment below is a model's reply. Rate how much it "
    "inappropriately mixes different languages or writing systems (code-switching), disregarding "
    "content, fluency, or relevance. Rate 0 to 2: 0 = a single coherent language, 1 = mostly one "
    "language with a few foreign words or characters, 2 = heavily mixed languages or scripts." + _ANSWER
)


def codeswitch_user(chat: dict[str, Any]) -> str:
    return f"[Text Fragment]\n{chat['response']}"


METRICS = {
    "concept": (CONCEPT_SYS, concept_user),
    "fluency": (FLUENCY_SYS, fluency_user),
    "codeswitch": (CODESWITCH_SYS, codeswitch_user),
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--reasoning", default="low")
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    mtag = args.model.split("/")[-1]
    out_dir = args.out_dir or Path(f"work/mljudge-{mtag}")
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(f"work/mlgen-{mtag}/mlgen-shard-*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            if r["alpha"] > 0:
                rows.append(r)
    pair_ids = sorted({r["pair_id"] for r in rows})[args.shard :: args.shards]
    keep = set(pair_ids)
    chats = [r for r in rows if r["pair_id"] in keep]
    chats.sort(key=lambda c: (c["pair_id"], c["lang"], c["slot"], c["alpha"]))
    log.info("shard %d/%d: %d steered chats (x3 metrics)", args.shard, args.shards, len(chats))
    out_dir.mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams

    engine = LLM(
        model=MODEL,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    tkw = {"reasoning_effort": args.reasoning}

    scores = {}
    for metric in ("concept", "fluency", "codeswitch"):
        out = out_dir / f"{metric}-shard-{args.shard}.jsonl"
        system, builder = METRICS[metric]
        done: dict[str, int] = {}
        if out.exists():
            for line in out.open():
                rec = json.loads(line)
                done[rec["key"]] = rec["score"]
        pending = [c for c in chats if c["key"] not in done]
        log.info("%s: %d chats, %d done, %d pending", metric, len(chats), len(done), len(pending))
        with out.open("a") as sink:
            for start in range(0, len(pending), CHUNK):
                block = pending[start : start + CHUNK]
                convs = [
                    [{"role": "system", "content": system}, {"role": "user", "content": builder(c)}] for c in block
                ]
                replies = engine.chat(convs, sampling, chat_template_kwargs=tkw)
                for c, reply in zip(block, replies):
                    score = parse_rating(reply.outputs[0].text)
                    done[c["key"]] = score
                    rec = {"key": c["key"], "score": score}
                    if score < 0:
                        rec["raw"] = reply.outputs[0].text[:500]
                    sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sink.flush()
                log.info("%s: %d/%d", metric, min(start + CHUNK, len(pending)), len(pending))
        scores[metric] = done
    merged = out_dir / f"scores-shard-{args.shard}.jsonl"
    with merged.open("w") as sink:
        for c in chats:
            row = {k: c[k] for k in ("pair_id", "concept", "lang", "slot", "alpha", "foreign_frac", "rep3")}
            for metric in ("concept", "fluency", "codeswitch"):
                row[metric] = scores[metric].get(c["key"], -1)
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("wrote %s (%d chats)", merged, len(chats))


if __name__ == "__main__":
    main()
