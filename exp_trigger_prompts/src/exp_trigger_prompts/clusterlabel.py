from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

MODEL = "openai/gpt-oss-120b"
SYS = (
    "You are analyzing an unsupervised clustering of behavioral and dispositional concepts derived "
    "from a language model's own internal representations. Each cluster is a set of concepts the "
    "model encodes along a similar direction. Your job is to name the shared axis the cluster "
    "represents, as the model appears to see it -- not to impose an external taxonomy. Some clusters "
    "are one crisp feature; some are a broad axis; some are genuine grab-bags. Report honestly.\n"
    "Your visible answer must be ONLY a single-line JSON object, nothing else, in exactly this form:\n"
    '{"label": "<2-5 word name of the shared axis or feature>", "description": "<one sentence>", '
    '"outliers": ["<member concepts that do not fit the shared theme>"], '
    '"coherence": <integer 1-5, 5 = one clear theme, 1 = incoherent grab-bag>}'
)
JSON_RE = re.compile(r"\{.*\}", re.S)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--resolution", default="2.0")
    ap.add_argument("--reasoning", default="medium")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = ap.parse_args()

    d = json.loads(Path("work/conceptclusters/clusters.json").read_text())
    concept = d["concept"]
    comms = sorted(d["resolutions"][args.resolution]["communities"], key=lambda c: -len(c["members"]))
    convs = []
    for c in comms:
        names = [concept[i] for i in c["members"]]
        user = (
            f"Here are {len(names)} concepts the model groups into one cluster:\n"
            + "\n".join(f"- {x}" for x in names)
            + "\n\nName the shared axis and return ONLY the JSON object."
        )
        convs.append([{"role": "system", "content": SYS}, {"role": "user", "content": user}])

    from vllm import LLM, SamplingParams

    engine = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        max_model_len=8192,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    replies = engine.chat(convs, sampling, chat_template_kwargs={"reasoning_effort": args.reasoning})

    out = []
    for c, reply in zip(comms, replies, strict=True):
        text = reply.outputs[0].text
        m = JSON_RE.search(text)
        parsed: dict[str, Any]
        try:
            parsed = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            parsed = {}
        out.append(
            {
                "members": c["members"],
                "concepts": [concept[i] for i in c["members"]],
                "size": len(c["members"]),
                "tightness": c["tightness"],
                "llama_tight": c["llama_tight"],
                "reliab": c["reliab"],
                "label": parsed.get("label", "?"),
                "description": parsed.get("description", ""),
                "outliers": parsed.get("outliers", []),
                "coherence": parsed.get("coherence"),
                "raw": None if parsed else text[:600],
            }
        )
        print(
            f"[{len(c['members']):3d}] coh={parsed.get('coherence')} "
            f"tq={c['tightness']:.2f}  {parsed.get('label', '?')}",
            flush=True,
        )

    Path("work/clusterlabel").mkdir(parents=True, exist_ok=True)
    Path(f"work/clusterlabel/labels-res{args.resolution}.json").write_text(
        json.dumps({"resolution": args.resolution, "communities": out}, ensure_ascii=False, indent=1)
    )
    print(f"\nwrote work/clusterlabel/labels-res{args.resolution}.json")


if __name__ == "__main__":
    main()
