from __future__ import annotations

import json
import logging
import re
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from .sources import LMSYS

log = logging.getLogger("judge")
ANS_RE = re.compile(r"<ANS>\s*(yes|no)\s*</ANS>", re.IGNORECASE)
ANS_HL_RE = re.compile(r"<ANS>\s*(1|2|incidental)\s*</ANS>", re.IGNORECASE)

SYSTEM_HL = (
    "You are a precise annotator for a mechanistic-interpretability dataset. You are shown a real "
    "conversation in which an automatic concept detector's strongest firing has been marked: a few spans "
    "wrapped in << >> and a few in [[ ]]. Two opposed concepts are named; the prompt says which concept "
    "each bracket type fired toward. Give ONE overall verdict for the marking as a whole -- do not label "
    "spans one by one. Some marked spans will be generic words or punctuation; IGNORE those and weigh the "
    "marked spans that carry real meaning. If those meaningful marks point to the first concept, answer 1; "
    "to the second concept, answer 2. If the marks are ONLY generic words or punctuation with nothing "
    "specific to either concept (the detector fired on surface form, not meaning), answer incidental. "
    "Judge in any language. Keep your reasoning brief, then end with EXACTLY one tag: "
    "<ANS>1</ANS>, <ANS>2</ANS>, or <ANS>incidental</ANS>."
)


def hl_prompt(c1: str, c2: str, cls: str, open_name: str, close_name: str, u: str, a: str) -> str:
    return (
        f"Concept 1: {c1}\nConcept 2: {c2}\nCLASS: {cls}\n\n"
        f"In the conversation below, << >> marks spans that fired toward {open_name}, and "
        f"[[ ]] marks spans that fired toward {close_name}.\n\n"
        f"CONVERSATION:\nUser: {u}\nAssistant: {a}\n\n"
        f"Taking the marked spans together and ignoring any that are just generic words or "
        f"punctuation, do the meaningful ones genuinely express Concept 1, Concept 2, or neither "
        f"(incidental)? Give one overall verdict."
    )


ANS_CLASS_RE = re.compile(r"<ANS>\s*(0|1|2)\s*</ANS>")

SYSTEM_CLASS = (
    "You are a precise annotator. You are shown a real conversation (a user message and the assistant's "
    "reply) and ONE thematic CLASS from an ontology -- a family of opposed concept pairs. Rate how much the "
    "conversation ENGAGES this class's theme, on a 0-2 scale: 0 = unrelated, the theme does not appear; "
    "1 = the theme is partially or tangentially present; 2 = the theme is clearly and centrally present. "
    "Judge the content in any language. Keep your reasoning brief, then end with EXACTLY one tag: "
    "<ANS>0</ANS>, <ANS>1</ANS>, or <ANS>2</ANS>."
)


def class_prompt(u: str, a: str, name: str, pairs: list[tuple[str, str]], cap: int) -> str:
    listing = "\n".join(f"- {c} || {ant}" for c, ant in pairs)
    return (
        f"CONVERSATION:\nUser: {u[:cap]}\nAssistant: {a[:cap]}\n\n"
        f"---\nCLASS: {name}\nThis class is the axis spanning these opposed concept pairs:\n{listing}\n\n"
        f"How much does this conversation engage the theme of this class? Rate 0, 1, or 2."
    )


def run_classcov(args: Any) -> None:
    from collections import OrderedDict

    convs = json.loads((args.work / "classcov_convs.json").read_text())
    features = json.loads(args.features.read_text())["features"]
    classes: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
    for f in features:
        classes.setdefault(f["class_name"], []).append((f["concept"], f["antagonist"]))
    my_convs = convs[args.shard :: args.shards]
    prompts: list[Any] = []
    index: list[tuple[str, str]] = []
    for cv in my_convs:
        for name, pairs in classes.items():
            prompts.append(
                [
                    {"role": "system", "content": SYSTEM_CLASS},
                    {"role": "user", "content": class_prompt(cv["user"], cv["assistant"], name, pairs, args.cap)},
                ]
            )
            index.append((cv["conversation_id"], name))
    log.info(
        "classcov shard %d/%d: %d conversations x %d classes = %d calls",
        args.shard,
        args.shards,
        len(my_convs),
        len(classes),
        len(prompts),
    )

    from vllm import LLM, SamplingParams

    engine = LLM(
        model=args.model,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    replies = engine.chat(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        chat_template_kwargs={"reasoning_effort": args.effort},
    )
    out = []
    for (cid, name), reply in zip(index, replies):
        text = reply.outputs[0].text
        m = ANS_CLASS_RE.findall(text)
        out.append(
            {
                "conversation_id": cid,
                "class": name,
                "score": (int(m[-1]) if m else -1),
                "reason": ANS_CLASS_RE.sub("", text).strip()[:120],
            }
        )
    (args.work / f"classcov-{args.shard}.json").write_text(json.dumps(out, ensure_ascii=False))
    log.info("classcov shard %d done: %d judged -> classcov-%d.json", args.shard, len(out), args.shard)


def run_study(args: Any) -> None:
    import hashlib

    recs = json.loads((args.work / f"study_tagged-{args.shard}.json").read_text())
    prompts: list[Any] = []
    for r in recs:
        flip = int(hashlib.md5(f"{r['conversation_id']}:{r['pair_id']}".encode()).hexdigest(), 16) & 1
        concept, antag = r["concept"], r["antagonist"]
        c1, c2 = (antag, concept) if flip else (concept, antag)
        prompts.append(
            [
                {"role": "system", "content": SYSTEM_HL},
                {
                    "role": "user",
                    "content": hl_prompt(c1, c2, r["class"], concept, antag, r["tagged_user"], r["tagged_assistant"]),
                },
            ]
        )
    log.info(
        "study shard %d/%d: %d records (%d controls)",
        args.shard,
        args.shards,
        len(recs),
        sum(x["control"] for x in recs),
    )

    from vllm import LLM, SamplingParams

    engine = LLM(
        model=args.model,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    replies = engine.chat(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        chat_template_kwargs={"reasoning_effort": args.effort},
    )
    out = []
    for r, reply in zip(recs, replies):
        text = reply.outputs[0].text
        m = ANS_HL_RE.findall(text)
        ans = m[-1].lower() if m else "?"
        flip = int(hashlib.md5(f"{r['conversation_id']}:{r['pair_id']}".encode()).hexdigest(), 16) & 1
        hl = (
            "incidental"
            if ans == "incidental"
            else ("antagonist" if (ans == "1") == bool(flip) else "concept")
            if ans in ("1", "2")
            else "?"
        )
        out.append(
            {k: r[k] for k in ("pair_id", "pole", "conversation_id", "plain", "bucket", "control")}
            | {"hl": hl, "reason": ANS_HL_RE.sub("", text).strip()[:600]}
        )
    (args.work / f"study_judged-{args.shard}.json").write_text(json.dumps(out, ensure_ascii=False))
    log.info("study shard %d done: %d judged -> study_judged-%d.json", args.shard, len(out), args.shard)


SYSTEM = (
    "You are a precise annotator for a mechanistic-interpretability dataset. You are given a CONCEPT "
    "(with a short description of its opposed pair) and a real conversation (a user message and the "
    "assistant's reply). Decide whether the conversation genuinely EXPRESSES or is ABOUT that concept "
    "-- not merely mentioning a related word, and not the opposite pole. Judge what is written, in any "
    "language. Keep your reasoning brief, then end with EXACTLY one tag: <ANS>yes</ANS> or <ANS>no</ANS>."
)


def user_prompt(target: str, opposed: str, cls: str, desc: str, user: str, asst: str, cap: int) -> str:
    return (
        f"CONCEPT: {target}   (opposed pole: {opposed})\nCLASS: {cls}\nDESCRIPTION: {desc}\n\n"
        f"CONVERSATION:\nUser: {user[:cap]}\nAssistant: {asst[:cap]}\n\n"
        f'Does this conversation genuinely express or center on "{target}" '
        f'(rather than "{opposed}", or being unrelated)?'
    )


SYSTEM_DOC = (
    "You are a precise annotator for a mechanistic-interpretability dataset. You are given a CONCEPT "
    "(with a short description of its opposed pair) and a passage of real web text. Decide whether the "
    "passage genuinely EXPRESSES or is ABOUT that concept -- not merely mentioning a related word, and "
    "not the opposite pole. Judge what is written, in any language. Keep your reasoning brief, then end "
    "with EXACTLY one tag: <ANS>yes</ANS> or <ANS>no</ANS>."
)


def doc_prompt(target: str, opposed: str, cls: str, desc: str, text: str, cap: int) -> str:
    return (
        f"CONCEPT: {target}   (opposed pole: {opposed})\nCLASS: {cls}\nDESCRIPTION: {desc}\n\n"
        f"TEXT:\n{text[:cap]}\n\n"
        f'Does this text genuinely express or center on "{target}" '
        f'(rather than "{opposed}", or being unrelated)?'
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--work", type=Path, default=Path("work"))
    parser.add_argument("--features", type=Path, default=Path("data/features.json"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--study", action="store_true", help="highlighted three-way judge over study_tagged-*")
    parser.add_argument("--corpus", default="lmsys", help="lmsys (conversations) or openwebtext (documents)")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--effort", default="low")
    parser.add_argument("--cap", type=int, default=1500)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    if args.study:
        run_study(args)
        return

    features = json.loads(args.features.read_text())["features"]
    if args.merge:
        parts: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"con": [], "ant": []})
        for f in sorted(args.work.glob("judged-*.json")):
            for r in json.loads(f.read_text()):
                parts[r["pair_id"]][r["pole"]].append(
                    {"conversation_id": r["conversation_id"], "verdict": r["verdict"], "reason": r["reason"]}
                )
        merged = [
            {
                "pair_id": p,
                "concept": ft["concept"],
                "antagonist": ft["antagonist"],
                "concept_yes": sum(c["verdict"] == "yes" for c in parts[p]["con"]),
                "antagonist_yes": sum(c["verdict"] == "yes" for c in parts[p]["ant"]),
                "con": parts[p]["con"],
                "ant": parts[p]["ant"],
            }
            for p, ft in enumerate(features)
        ]
        (args.work / "judged.json").write_text(json.dumps({"features": merged}, ensure_ascii=False))
        log.info("combined %d shards -> judged.json", len(list(args.work.glob("judged-*.json"))))
        return

    cand = json.loads((args.work / "candidates.json").read_text())["features"]
    jobs = [(f["pair_id"], pole, c["conversation_id"]) for f in cand for pole in ("con", "ant") for c in f[pole]]
    jobs = jobs[args.shard :: args.shards]
    wanted = {cid for _, _, cid in jobs}
    owt = args.corpus == "openwebtext"
    texts: dict[str, Any] = {}
    if owt:
        from datasets import load_dataset

        from .sources import OPENWEBTEXT

        ds = load_dataset(OPENWEBTEXT, split="train")
        for cid in wanted:
            try:
                texts[cid] = ds[int(cid.split("-", 1)[1])]["text"]
            except (ValueError, IndexError, KeyError):
                continue
    elif args.corpus == "alpaca":
        from datasets import load_dataset

        from .sources import ALPACA, alpaca_user

        ds = load_dataset(ALPACA, split="train")
        for cid in wanted:
            try:
                ex = ds[int(cid.split("-", 1)[1])]
                texts[cid] = (alpaca_user(ex["instruction"], ex["input"]), ex["output"])
            except (ValueError, IndexError, KeyError):
                continue
    else:
        root = Path(snapshot_download(LMSYS, repo_type="dataset"))
        for shard in sorted(root.rglob("*.parquet")):
            table = pq.read_table(shard, columns=["conversation_id", "conversation"])
            for cid, turns in zip(
                table.column("conversation_id").to_pylist(), table.column("conversation").to_pylist()
            ):
                if cid in wanted and cid not in texts:
                    oi = next((i for i, t in enumerate(turns) if t["role"] == "user"), None)
                    if oi is None:
                        continue
                    rep = next((t["content"] for t in turns[oi + 1 :] if t["role"] == "assistant"), "")
                    texts[cid] = (turns[oi]["content"], rep)
            if len(texts) == len(wanted):
                break
    system = SYSTEM_DOC if owt else SYSTEM
    prompts: list[Any] = []
    index: list[tuple[int, str, str]] = []
    for pid, pole, cid in jobs:
        if cid not in texts:
            continue
        ft = features[pid]
        target, opposed = (ft["concept"], ft["antagonist"]) if pole == "con" else (ft["antagonist"], ft["concept"])
        if owt:
            content = doc_prompt(target, opposed, ft["class_name"], ft["narrative_guidance"], texts[cid], args.cap)
        else:
            u, a = texts[cid]
            content = user_prompt(target, opposed, ft["class_name"], ft["narrative_guidance"], u, a, args.cap)
        prompts.append([{"role": "system", "content": system}, {"role": "user", "content": content}])
        index.append((pid, pole, cid))
    log.info("shard %d/%d: %d prompts (%d distinct conversations)", args.shard, args.shards, len(prompts), len(wanted))

    from vllm import LLM, SamplingParams

    engine = LLM(
        model=args.model,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    replies = engine.chat(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        chat_template_kwargs={"reasoning_effort": args.effort},
    )
    out = []
    for (pid, pole, cid), reply in zip(index, replies):
        text = reply.outputs[0].text
        m = ANS_RE.findall(text)
        out.append(
            {
                "pair_id": pid,
                "pole": pole,
                "conversation_id": cid,
                "verdict": (m[-1].lower() if m else "?"),
                "reason": ANS_RE.sub("", text).strip()[:600],
            }
        )
    (args.work / f"judged-{args.shard}.json").write_text(json.dumps(out, ensure_ascii=False))
    yes = sum(r["verdict"] == "yes" for r in out)
    log.info("shard %d done: %d judged, %d yes -> judged-%d.json", args.shard, len(out), yes, args.shard)


if __name__ == "__main__":
    main()
