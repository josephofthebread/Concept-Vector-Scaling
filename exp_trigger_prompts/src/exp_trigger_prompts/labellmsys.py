from __future__ import annotations

import json
import logging
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

log = logging.getLogger("labellmsys")

CLASS_RE = re.compile(r"<CLASS>\s*(-?\d+)\s*</CLASS>", re.IGNORECASE)
CONF_RE = re.compile(r"<CONF>\s*([0-9]*\.?[0-9]+)\s*</CONF>", re.IGNORECASE)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--sample", type=Path, default=Path("work/lmsys-auroc/sample.json"))
    parser.add_argument("--ontology", type=Path, default=Path("data/ontology.json"))
    parser.add_argument("--out", type=Path, default=Path("work/lmsys-auroc/sample_gptoss.json"))
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--tensor-parallel", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=640)
    parser.add_argument("--effort", default="low")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    classes = json.loads(args.ontology.read_text())["classes"]
    n_cls = len(classes)
    taxonomy = "\n".join(
        [
            f"{i}\t{c['name']} — e.g. "
            + "; ".join(f"{p['concept']} || {p['antagonist']}" for p in c.get("pairs", [])[:3])
            for i, c in enumerate(classes)
        ]
    )
    sample = json.loads(args.sample.read_text())
    log.info("%d classes, %d conversations to label", n_cls, len(sample))

    system = (
        "You are a meticulous taxonomist. Below is a numbered list of 148 concept classes, each a "
        "family of opposed concept pairs. You will be given the first user message of a conversation. "
        "Decide which SINGLE class best captures the conversation's dominant concept or dichotomy — "
        "the axis the message is really about, not a superficial keyword. If none of the classes "
        "genuinely fit, answer -1 (do not force a weak match).\n\n"
        f"CLASSES (index<TAB>name — example poles):\n{taxonomy}\n\n"
        "Answer with your reasoning kept minimal, then end with EXACTLY two tags on the final line:\n"
        "<CLASS>INDEX</CLASS><CONF>P</CONF>\n"
        "where INDEX is the integer class index (0.." + str(n_cls - 1) + ") or -1, and P is your "
        "confidence in [0,1]. Output nothing after the tags."
    )

    from vllm import LLM, SamplingParams

    engine = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    prompts: list[Any] = [
        [{"role": "system", "content": system}, {"role": "user", "content": f"First user message:\n{it['user']}"}]
        for it in sample
    ]
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    log.info("labelling %d conversations (effort=%s, tp=%d)", len(prompts), args.effort, args.tensor_parallel)
    replies = engine.chat(prompts, sampling, chat_template_kwargs={"reasoning_effort": args.effort})

    out = []
    labelled = abstain = bad = 0
    per_class: dict[int, int] = {}
    for it, reply in zip(sample, replies):
        text = reply.outputs[0].text
        matches = CLASS_RE.findall(text)
        if not matches:
            cid, conf = -1, 0.0
        else:
            try:
                cid = int(matches[-1])
            except ValueError:
                cid, conf = -1, 0.0
            else:
                conf_m = CONF_RE.findall(text)
                conf = float(conf_m[-1]) if conf_m else 0.0
        if cid < -1 or cid >= n_cls:
            cid, bad = -1, bad + 1
        if cid == -1:
            abstain += 1
            name = ""
        else:
            labelled += 1
            per_class[cid] = per_class.get(cid, 0) + 1
            name = classes[cid]["name"]
        out.append({"conversation_id": it["conversation_id"], "class_id": cid, "class_name": name, "confidence": conf})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    covered = sorted(per_class.values())
    median = covered[len(covered) // 2] if covered else 0
    log.info(
        "wrote %s: %d labelled, %d abstain, %d out-of-range; %d classes covered, median %d/class",
        args.out,
        labelled,
        abstain,
        bad,
        len(per_class),
        median,
    )


if __name__ == "__main__":
    main()
