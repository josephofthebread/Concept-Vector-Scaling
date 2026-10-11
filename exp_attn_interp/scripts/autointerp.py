from __future__ import annotations

import json
import logging
import re
from argparse import ArgumentParser
from pathlib import Path

log = logging.getLogger("autointerp")
MODEL = "openai/gpt-oss-120b"
SYSTEM = (
    "You are labeling features of a language model. Each feature fires on certain tokens. You are shown the tokens on which "
    "a feature fired most strongly, each marked ⟦like this⟧ inside a short left context, taken from short stories. Give the "
    "feature a label of at most 6 words that says what the marked tokens have in common (a word or word family, a "
    "grammatical role, a topic, a stance or attitude, a language, punctuation or formatting). If no pattern is visible say "
    '"unclear". Your visible answer must be ONLY: "Label: [[your label]]".'
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = ArgumentParser()
    p.add_argument("--inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-per-row", type=int, default=400)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    data = json.loads(args.inp.read_text())
    items = []
    for r, rd in data["rows"].items():
        feats = list(rd["contexts"].items())[: args.max_per_row]
        for j, ctxs in feats:
            shown = "\n".join(f"- {c['ctx']}" for c in ctxs[:12])
            items.append({"row": int(r), "feature": int(j), "prompt": f"[Tokens the feature fired on]\n{shown}"})
    if args.limit:
        items = items[: args.limit]
    done = {}
    if args.out.exists():
        for l in args.out.open():
            d = json.loads(l)
            if d["label"] != "?":
                done[(d["row"], d["feature"])] = d["label"]
    pending = [it for it in items if (it["row"], it["feature"]) not in done]
    log.info("%d features, %d done, %d pending", len(items), len(done), len(pending))
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=MODEL, tensor_parallel_size=1, max_model_len=4096, gpu_memory_utilization=0.9, enable_prefix_caching=True
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=512)
    with args.out.open("a") as sink:
        for start in range(0, len(pending), 2048):
            block = pending[start : start + 2048]
            convs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": it["prompt"]}] for it in block]
            replies = engine.chat(convs, sampling, chat_template_kwargs={"reasoning_effort": "low"})
            for it, rep in zip(block, replies):
                text = rep.outputs[0].text
                tail = text.split("final")[-1] if "final" in text else text
                m = re.findall(r"Label:\s*\[*\s*([^\[\]\n]+?)\s*\]*\s*$", tail.strip())
                if m:
                    label = m[-1].strip().strip("\"'")
                else:
                    m = re.findall(r"Label:\s*\[*\s*([^\[\]\n]+)", tail)
                    label = m[-1].strip().strip("\"'").rstrip("]") if m else "?"
                sink.write(
                    json.dumps({"row": it["row"], "feature": it["feature"], "label": label, "raw": text[-200:]}) + "\n"
                )
            sink.flush()
            log.info("%d/%d", min(start + 2048, len(pending)), len(pending))
    log.info("done")


if __name__ == "__main__":
    main()
