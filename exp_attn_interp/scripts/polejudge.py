from __future__ import annotations

import json
import logging
import re
from argparse import ArgumentParser
from pathlib import Path

log = logging.getLogger("polejudge")
MODEL = "openai/gpt-oss-120b"
SYSTEM = (
    "Please act as an impartial judge. You are given a contrast between two opposite behaviors or stances, A and B, "
    "with a short description of the dichotomy, and a text fragment that a model wrote in reply to a prompt. Decide "
    "which side the text enacts or expresses MORE, judging what the text does and says, not what it mentions. "
    "Reply A if the text leans to side A, B if it leans to side B, N if it expresses neither side or both equally. "
    "Your visible answer must be ONLY the letter, on one line, in exactly this format and nothing else: "
    '"Answer: [[X]]" where X is A, B, or N.'
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = ArgumentParser()
    p.add_argument("--inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--features", type=Path, default=Path("data/features.json"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--chunk", type=int, default=2048)
    args = p.parse_args()
    feats = {f["pair_id"]: f for f in json.loads(args.features.read_text())["features"]}
    chats = [json.loads(l) for l in args.inp.open()]
    if args.limit:
        chats = chats[: args.limit]
    done = {}
    if args.out.exists():
        for l in args.out.open():
            r = json.loads(l)
            if r["answer"] in ("A", "B", "N"):
                done[r["key"]] = r["answer"]
    pending = [c for c in chats if c["key"] not in done]
    log.info("%d chats, %d done, %d pending", len(chats), len(done), len(pending))
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=MODEL, tensor_parallel_size=1, max_model_len=4096, gpu_memory_utilization=0.9, enable_prefix_caching=True
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=1024)
    with args.out.open("a") as sink:
        for start in range(0, len(pending), args.chunk):
            block = pending[start : start + args.chunk]
            convs = []
            for c in block:
                f = feats[int(c["pair_id"])]
                convs.append(
                    [
                        {"role": "system", "content": SYSTEM},
                        {
                            "role": "user",
                            "content": (
                                f'[Contrast]\nA = "{f["concept"]}"\nB = "{f["antagonist"]}"\n{f["narrative_guidance"]}\n\n'
                                f"[Prompt the text replied to]\n{c['prompt']}\n\n[Text Fragment]\n{c['response'][:2500]}"
                            ),
                        },
                    ]
                )
            replies = engine.chat(convs, sampling, chat_template_kwargs={"reasoning_effort": "low"})
            for c, rep in zip(block, replies):
                text = rep.outputs[0].text
                tail = text.split("final")[-1] if "final" in text else text
                ans = "?"
                for pat in (r"Answer:\s*\[+\s*([ABN])\s*\]+", r"\[+\s*([ABN])\s*\]+", r"Answer:\s*([ABN])\b"):
                    m = re.findall(pat, tail)
                    if m:
                        ans = m[-1]
                        break
                rec = {"key": c["key"], "pair_id": c["pair_id"], "arm": c["arm"], "alpha": c["alpha"], "answer": ans}
                rec["raw"] = text[-400:]
                sink.write(json.dumps(rec) + "\n")
            sink.flush()
            log.info("%d/%d", min(start + args.chunk, len(pending)), len(pending))
    log.info("done")


if __name__ == "__main__":
    main()
