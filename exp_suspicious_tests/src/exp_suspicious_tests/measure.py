from __future__ import annotations

import json
import logging
import time
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from transformers import AutoModel

from exp_suspicious_tests.config import (
    INSTRUCT_CHECKPOINTS,
    INSTRUCT_MODEL,
    PAIRS,
    POSITIONS,
    attention_implementation,
)
from exp_suspicious_tests.vectors import load_vectors_for

log = logging.getLogger("measure")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL, choices=INSTRUCT_CHECKPOINTS)
    parser.add_argument("--presentation", default="assistant")
    parser.add_argument("--transcripts", type=Path, default=Path("work/transcripts.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", default="", help="filter: main|roleplay")
    parser.add_argument("--probe", default="", help="filter: scared|angry|sad|happy|calm")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model = AutoModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=attention_implementation(args.model, need_attentions=False),
        local_files_only=args.offline,
        device_map={"": "cuda"},
    ).eval()
    vectors = load_vectors_for(args.model, args.presentation)["diff"]
    layers = vectors.shape[0]
    rows = []
    with args.transcripts.open() as stream:
        for line in stream:
            row = json.loads(line)
            if args.arm and row["arm"] != args.arm:
                continue
            if args.probe and row["probe"] != args.probe:
                continue
            rows.append(row)
    rows.sort(key=lambda r: (r["level"], r["label"], r["seed"], r["arm"], r["probe"]))
    rows = rows[: args.limit] if args.limit else rows
    if not rows:
        raise SystemExit("no transcripts matched the filter")

    proj = torch.zeros(len(rows), len(POSITIONS), layers, PAIRS)
    meta: list[dict[str, Any]] = []
    started = time.monotonic()
    for number, row in enumerate(rows):
        ids = torch.tensor([row["input_ids"]], device="cuda")
        with torch.inference_mode():
            output = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        xs = {
            "emotion": int(row["emotion_position"]),
            "assistant": int(row["assistant_position"]),
            "answer": int(row["answer_position"]),
        }
        for position_index, name in enumerate(POSITIONS):
            x = xs[name]
            if x < 0 or x >= ids.shape[1]:
                proj[number, position_index] = torch.nan
                continue
            stacked = torch.stack([output.hidden_states[layer][0, x].to(torch.float32) for layer in range(layers)])
            proj[number, position_index] = torch.einsum("lh,lkh->lk", stacked, vectors).cpu()
        meta.append({key: row[key] for key in ("level", "label", "seed", "arm", "probe", "answer")})
        if (number + 1) % 50 == 0 or number + 1 == len(rows):
            rate = (number + 1) / (time.monotonic() - started)
            log.info("%s/%s transcripts, %.1f min left", number + 1, len(rows), (len(rows) - number - 1) / rate / 60)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"proj": proj},
        str(args.out),
        metadata={
            "meta": json.dumps(meta),
            "positions": json.dumps(POSITIONS),
            "construction": "diff",
            "model": args.model,
            "presentation": args.presentation,
        },
    )
    log.info("wrote %s (%.2f GiB, %s trials)", args.out, args.out.stat().st_size / 2**30, len(meta))


if __name__ == "__main__":
    main()
