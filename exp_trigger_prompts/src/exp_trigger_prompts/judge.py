from __future__ import annotations

import collections
import json
import logging
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

log = logging.getLogger("trigger-judge")

MODEL = "openai/gpt-oss-120b"
SCALE = 2
CHUNK = 8192
RATING_RE = re.compile(r"\[\[\s*([0-9])\s*\]\]")
RATING_FALLBACK = re.compile(r"(?:Rating|rating)\D{0,12}([0-2])")

_ANSWER = (
    " Your visible answer must be ONLY the rating, on one line, in exactly this format and nothing "
    'else: "Rating: [[N]]" where N is 0, 1, or 2.'
)
CONCEPT_SYS = (
    "Please act as an impartial judge and evaluate whether the specified concept is expressed in the "
    "sentence fragment below. Focus solely on whether the concept is clearly incorporated, without "
    "regard for grammar or coherence. Rate its presence 0 to 2: 0 = not present at all, 1 = somewhat "
    "present but minimal or awkward, 2 = fully and naturally incorporated." + _ANSWER
)
INSTRUCT_SYS = (
    "Please act as an impartial judge and evaluate whether the sentence fragment below is related to "
    "the instruction. Focus solely on topical relatedness, regardless of grammar, coherence, or "
    "informativeness. Rate the relatedness 0 to 2: 0 = unrelated, 1 = somewhat or indirectly related, "
    "2 = clearly and directly related." + _ANSWER
)
FLUENCY_SYS = (
    "Please act as an impartial judge and evaluate the fluency of the sentence fragment below. Focus "
    "solely on fluency, disregarding completeness, relevance, or informativeness. Rate fluency 0 to 2: "
    "0 = not fluent / highly unnatural (incomprehensible or repetitive), 1 = somewhat fluent with "
    "noticeable errors, 2 = fluent and almost perfect." + _ANSWER
)


def concept_user(chat: dict[str, Any]) -> str:
    return (
        f'[Concept]\n"{chat["pole"]}" -- the "{chat["pole"]}" side of the axis '
        f'"{chat["concept"]} vs {chat["antagonist"]}". {chat["guidance"]}\n\n'
        f"[Sentence Fragment]\n{chat['response']}"
    )


def instruct_user(chat: dict[str, Any]) -> str:
    return f"[Instruction]\n{chat['prompt']}\n\n[Sentence Fragment]\n{chat['response']}"


def fluency_user(chat: dict[str, Any]) -> str:
    return f"[Sentence Fragment]\n{chat['response']}"


METRICS = {
    "concept": (CONCEPT_SYS, concept_user),
    "instruct": (INSTRUCT_SYS, instruct_user),
    "fluency": (FLUENCY_SYS, fluency_user),
}


def parse_rating(text: str) -> int:
    """Rating from the reply: prefer [[n]], fall back to 'Rating ... n'; -1 if none. Clamped 0..SCALE."""
    for pat in (RATING_RE, RATING_FALLBACK):
        m = pat.findall(text)
        if m:
            return max(0, min(SCALE, int(m[-1])))
    return -1


def sampled_slots(slots: list[int], keep: int) -> set[int]:
    """`keep` evenly-spaced slots (distinct scenario shells) out of the sorted list."""
    ordered = sorted(set(slots))
    if len(ordered) <= keep:
        return set(ordered)
    step = len(ordered) / keep
    return {ordered[int(i * step)] for i in range(keep)}


def score_metric(
    engine: Any, sampling: Any, template_kwargs: dict[str, Any], metric: str, chats: list[dict[str, Any]], out: Path
) -> dict[str, int]:
    """Run one resumable, chunked metric pass over `chats` and return {key: score}."""
    system, builder = METRICS[metric]
    done: dict[str, int] = {}
    if out.exists():
        with out.open() as handle:
            for line in handle:
                rec = json.loads(line)
                done[rec["key"]] = rec["score"]
    pending = [c for c in chats if c["key"] not in done]
    log.info("%s: %d chats, %d done, %d pending", metric, len(chats), len(done), len(pending))
    with out.open("a") as sink:
        for start in range(0, len(pending), CHUNK):
            block = pending[start : start + CHUNK]
            convs = [[{"role": "system", "content": system}, {"role": "user", "content": builder(c)}] for c in block]
            replies = engine.chat(convs, sampling, chat_template_kwargs=template_kwargs)
            for c, reply in zip(block, replies):
                text = reply.outputs[0].text
                score = parse_rating(text)
                done[c["key"]] = score
                rec: dict[str, Any] = {"key": c["key"], "score": score}
                if score < 0:
                    rec["raw"] = text[:500]
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sink.flush()
            log.info("%s: %d/%d", metric, min(start + CHUNK, len(pending)), len(pending))
    return done


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--steered", type=Path, required=True, help="merged sweep jsonl")
    parser.add_argument("--features", type=Path, default=Path("data/features.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--prompts", type=int, default=8, help="prompts kept per concept (evenly spaced)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--reasoning", default="low")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0, help="cap chats for a test run (0 = all)")
    args = parser.parse_args()

    onto = {f["pair_id"]: f for f in json.loads(args.features.read_text())["features"]}
    by_pair_slots: dict[int, set[int]] = collections.defaultdict(set)
    rows: list[dict[str, Any]] = []
    for line in args.steered.open():
        r = json.loads(line)
        if r["alpha"] == 0.0:
            continue
        by_pair_slots[r["pair_id"]].add(r["slot"])
        rows.append(r)
    keep_slots = {p: sampled_slots(list(s), args.prompts) for p, s in by_pair_slots.items()}
    pair_ids = sorted(by_pair_slots)[args.shard :: args.shards]
    keep_pairs = set(pair_ids)
    chats: list[dict[str, Any]] = []
    for r in rows:
        pid = r["pair_id"]
        if pid not in keep_pairs or r["slot"] not in keep_slots[pid]:
            continue
        f = onto[pid]
        pole = f["concept"] if r["alpha"] > 0 else f["antagonist"]
        chats.append(
            {
                "key": f"{r['pair_id']}:{r['slot']}:L{r['layer']}:a{r['alpha']:g}",
                "pair_id": pid,
                "slot": r["slot"],
                "layer": r["layer"],
                "alpha": r["alpha"],
                "pole": pole,
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
    concepts = len({c["pair_id"] for c in chats})
    log.info("shard %d/%d: %d concepts, %d chats (x3 metrics)", args.shard, args.shards, concepts, len(chats))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams

    engine = LLM(
        model=MODEL,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    template_kwargs = {"reasoning_effort": args.reasoning}

    scores = {}
    for metric in ("concept", "instruct", "fluency"):
        scores[metric] = score_metric(
            engine, sampling, template_kwargs, metric, chats, args.out_dir / f"{metric}-shard-{args.shard}.jsonl"
        )

    merged = args.out_dir / f"scores-shard-{args.shard}.jsonl"
    unparsed = 0
    with merged.open("w") as sink:
        for c in chats:
            row = {k: c[k] for k in ("pair_id", "slot", "layer", "alpha")}
            for metric in ("concept", "instruct", "fluency"):
                row[metric] = scores[metric].get(c["key"], -1)
                unparsed += row[metric] < 0
            sink.write(json.dumps(row) + "\n")
    log.info("wrote %s (%d chats, %d unparsed sub-scores)", merged, len(chats), unparsed)


if __name__ == "__main__":
    main()
