from __future__ import annotations

import json
import logging
import time
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp_suspicious_tests.config import (
    BASE_MODEL,
    BASE_TEMPLATE,
    CHECKPOINTS,
    CONSTRUCTIONS,
    FAMILY_MARKERS,
    INSTRUCT_MODEL,
    LETTERS,
    MCQA_INSTRUCTION,
    PAIRS,
    attention_implementation,
    model_family,
)
from exp_suspicious_tests.vectors import load_vectors_for

log = logging.getLogger("mcqa")
MCQA_POSITIONS = ("last_prompt", "answer_letter", "assistant")


def build_prompt(tokenizer: Any, item: dict[str, Any], instruct: bool) -> list[int]:
    a, b, c, d = item["options"]
    if instruct:
        content = f"{item['question']}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\n{MCQA_INSTRUCTION}"
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ##
        # NOTE: the "Answer:" prefill forces the letter position; without it pilot 1.5 saw
        # non-compliant replies and letter logits degenerate at p(A)~0.99.
        ##
        text += "Answer:"
    else:
        text = BASE_TEMPLATE.format(question=item["question"], a=a, b=b, c=c, d=d, instruction=MCQA_INSTRUCTION)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", choices=("instruct", "base"), default="instruct")
    parser.add_argument(
        "--checkpoint",
        default="",
        choices=[""] + sorted(CHECKPOINTS),
        help="explicit checkpoint; overrides --model's Qwen default",
    )
    parser.add_argument("--items", type=Path, default=Path("data/mcqa.parquet"))
    parser.add_argument("--out", type=Path, default=Path("work/mcqa/out.safetensors"))
    parser.add_argument("--pilot", type=int, default=0, help="experiment 1.5: N items, print report")
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    checkpoint = args.checkpoint or (INSTRUCT_MODEL if args.model == "instruct" else BASE_MODEL)
    instruct = CHECKPOINTS[checkpoint].instruct
    markers = FAMILY_MARKERS[model_family(checkpoint)]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=args.offline, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        attn_implementation=attention_implementation(checkpoint, need_attentions=False),
        local_files_only=args.offline,
        device_map={"": "cuda"},
    ).eval()
    options = []
    for variant in [f" {letter}" for letter in LETTERS]:
        encoded = tokenizer(variant, add_special_tokens=False)["input_ids"]
        if len(encoded) != 1:
            raise RuntimeError(f"option letter {variant!r} is not a single token")
        options.append(encoded[0])
    tags = ["assistant"] if instruct else ["raw", "named_raw"]
    constructions = CONSTRUCTIONS if checkpoint in (INSTRUCT_MODEL, BASE_MODEL) else ("diff",)
    vectors = {tag: load_vectors_for(checkpoint, tag) for tag in tags}
    layers = int(model.config.num_hidden_layers) + 1

    items = pq.read_table(args.items).to_pylist()
    if args.pilot:
        mmlu = [item for item in items if item["source"] == "mmlu"][: args.pilot // 2]
        crafted = [item for item in items if item["source"] != "mmlu"][: args.pilot - len(mmlu)]
        compliant = 0
        for item in mmlu + crafted:
            ids = build_prompt(tokenizer, item, instruct)
            batch = torch.tensor([ids], device="cuda")
            with torch.inference_mode():
                generated = model.generate(
                    batch,
                    attention_mask=torch.ones_like(batch),
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                logits = model(input_ids=batch, use_cache=False).logits[0, -1, options].to(torch.float32)
            probabilities = torch.softmax(logits, dim=0)
            pilot_entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum().item()
            text = tokenizer.decode(generated[0, batch.shape[1] :], skip_special_tokens=True)
            letter = text.strip()[:1].upper()
            ok = letter in LETTERS
            compliant += ok
            print(
                f"[{item['source']:>12}] entropy={pilot_entropy:.3f} letters={probabilities.tolist()} "
                f"compliant={ok} reply={text.strip()!r} q={item['question'][:70]!r}",
                flush=True,
            )
        total = len(mmlu) + len(crafted)
        print(f"compliance: {compliant}/{total} ({'instruct' if instruct else 'base'})", flush=True)
        return

    proj = torch.full((len(items), len(tags), len(MCQA_POSITIONS), len(constructions), layers, PAIRS), torch.nan)
    entropy = torch.zeros(len(items))
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, item in enumerate(items):
        ids = build_prompt(tokenizer, item, instruct)
        batch = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            first = model(input_ids=batch, output_hidden_states=True, use_cache=False)
        logits = first.logits[0, -1, options].to(torch.float32)
        probabilities = torch.softmax(logits, dim=0)
        entropy[index] = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        chosen = int(probabilities.argmax().item())
        with torch.inference_mode():
            greedy = model.generate(
                batch,
                attention_mask=torch.ones_like(batch),
                max_new_tokens=6,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        greedy_text = tokenizer.decode(greedy[0, batch.shape[1] :], skip_special_tokens=True)
        written = next((ch for ch in greedy_text if ch in LETTERS), "")

        with torch.inference_mode():
            second = model(
                input_ids=torch.tensor([ids + [options[chosen]]], device="cuda"),
                output_hidden_states=True,
                use_cache=False,
            )
        positions = {"last_prompt": (first, len(ids) - 1), "answer_letter": (second, len(ids))}
        if instruct:
            token = tokenizer(markers["assistant_word"], add_special_tokens=False)["input_ids"][0]
            positions["assistant"] = (first, len(ids) - 1 - ids[::-1].index(token))
        for tag_index, tag in enumerate(tags):
            for position_index, name in enumerate(MCQA_POSITIONS):
                if name not in positions:
                    continue
                output, position = positions[name]
                stacked = torch.stack(
                    [output.hidden_states[layer][0, position].to(torch.float32) for layer in range(layers)]
                )
                for construction_index, construction in enumerate(constructions):
                    proj[index, tag_index, position_index, construction_index] = torch.einsum(
                        "lh,lkh->lk", stacked, vectors[tag][construction]
                    ).cpu()
        records.append(
            {
                "id": item["id"],
                "source": item["source"],
                "subject": item.get("subject", ""),
                "answer_index": item["answer_index"],
                "chosen": chosen,
                "correct": int(chosen == item["answer_index"]) if item["answer_index"] >= 0 else -1,
                "compliant": int(written == LETTERS[chosen]),
            }
        )
        if (index + 1) % 100 == 0 or index + 1 == len(items):
            rate = (index + 1) / (time.monotonic() - started)
            log.info("%s/%s items, %.1f min left", index + 1, len(items), (len(items) - index - 1) / rate / 60)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"proj": proj, "entropy": entropy},
        str(args.out),
        metadata={
            "records": json.dumps(records),
            "tags": json.dumps(tags),
            "positions": json.dumps(MCQA_POSITIONS),
            "constructions": json.dumps(constructions),
            "model": checkpoint,
        },
    )
    compliant = sum(r["compliant"] for r in records)
    log.info(
        "wrote %s (%.2f GiB); greedy-compliance %s/%s",
        args.out,
        args.out.stat().st_size / 2**30,
        compliant,
        len(records),
    )


if __name__ == "__main__":
    main()
