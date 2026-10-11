from __future__ import annotations

import json
import logging
import time
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp_suspicious_tests.chat import build_transcript, find_answer_position, render_ids
from exp_suspicious_tests.config import (
    ANSWER_TOKENS,
    ARMS,
    CHECKPOINTS,
    CONTINUATION_TOKENS,
    FAMILY_MARKERS,
    INSTRUCT_CHECKPOINTS,
    INSTRUCT_MODEL,
    PROBES,
    ROLEPLAY_ASSISTANT,
    ROLEPLAY_USER,
    SAMPLING,
    SEEDS,
    attention_implementation,
    generation_seed,
    model_family,
)

log = logging.getLogger("generate")


def load_messages(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text())
    records.sort(key=lambda r: (int(r["level"]), str(r["label"])))
    return records


def sample(model: Any, tokenizer: Any, input_ids: list[int], new_tokens: int, seed: int) -> list[int]:
    torch.manual_seed(seed)
    batch = torch.tensor([input_ids], device=model.device)
    with torch.inference_mode():
        output = model.generate(
            batch,
            attention_mask=torch.ones_like(batch),
            max_new_tokens=new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            **SAMPLING,
        )
    return output[0, batch.shape[1] :].tolist()


def continuation_for(model: Any, tokenizer: Any, hostile: str, arm: str, seed_index: int, cell: int) -> str:
    messages: list[dict[str, str]] = []
    if arm == "roleplay":
        messages += [
            {"role": "user", "content": ROLEPLAY_USER},
            {"role": "assistant", "content": ROLEPLAY_ASSISTANT},
        ]
    messages.append({"role": "user", "content": hostile})
    prompt_ids = render_ids(tokenizer, messages, add_generation_prompt=True)
    seed = generation_seed(cell, seed_index, 0 if arm == "main" else 1)
    generated = sample(model, tokenizer, prompt_ids, CONTINUATION_TOKENS, seed)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text.split("</think>")[-1].strip()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL, choices=INSTRUCT_CHECKPOINTS)
    parser.add_argument("--messages", type=Path, default=Path("data/trashtalk_messages.json"))
    parser.add_argument("--out", type=Path, default=Path("work/transcripts.jsonl"))
    parser.add_argument("--arm", choices=ARMS, default="", help="restrict to one arm (default: both)")
    parser.add_argument("--max-cells", type=int, default=0, help="bounded pilot; never a production artifact")
    parser.add_argument("--stride", type=int, default=1, help="take every Nth message (pilot: 5 = one per level)")
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    parser.add_argument(
        "--prefills",
        type=Path,
        default=None,
        help="JSON of fixed continuations (neutral prefill author); answers stay per-model",
    )
    parser.add_argument(
        "--prefill-model",
        default="",
        help="with --prefills-out: generate continuations with this neutral model and exit",
    )
    parser.add_argument("--prefills-out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.prefill_model or args.prefills_out:
        if not (args.prefill_model and args.prefills_out):
            raise SystemExit("--prefill-model and --prefills-out go together")
        ##
        # NOTE: deepseek-llm's tokenizer_config declares a Llama wrapper whose decode drops the
        # ByteLevel spaces; the generic fast tokenizer uses tokenizer.json's own decoder.
        ##
        from transformers import PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.prefill_model, local_files_only=args.offline)
        probe = "Of course, I am here to help you today."
        if tokenizer.decode(tokenizer(probe, add_special_tokens=False)["input_ids"]) != probe:
            raise SystemExit(f"{args.prefill_model}: tokenizer round-trip broken; refusing to write garbage")
        model = AutoModelForCausalLM.from_pretrained(
            args.prefill_model,
            dtype=torch.bfloat16,
            local_files_only=args.offline,
            device_map={"": "cuda"},
        ).eval()
        messages = load_messages(args.messages)
        prefills: dict[str, str] = {}
        for cell, record in enumerate(messages):
            for seed_index in SEEDS:
                for arm in ARMS:
                    text = continuation_for(model, tokenizer, record["text"], arm, seed_index, cell)
                    prefills[f"{record['level']}-{record['label']}-{seed_index}-{arm}"] = text
                    log.info("%s chars for %s", len(text), f"{record['level']}-{record['label']}-{seed_index}-{arm}")
        args.prefills_out.parent.mkdir(parents=True, exist_ok=True)
        args.prefills_out.write_text(
            json.dumps({"model": args.prefill_model, "prefills": prefills}, ensure_ascii=False, indent=1)
        )
        log.info("wrote %s continuations to %s", len(prefills), args.prefills_out)
        return

    if not CHECKPOINTS[args.model].instruct:
        raise SystemExit("trashtalk conversations need a chat model")
    markers = FAMILY_MARKERS[model_family(args.model)]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.offline, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=attention_implementation(args.model, need_attentions=False),
        local_files_only=args.offline,
        device_map={"": "cuda"},
    ).eval()

    prefills = {}
    if args.prefills:
        payload = json.loads(args.prefills.read_text())
        prefills = payload["prefills"]
        log.info("using %s fixed prefills from %s (%s)", len(prefills), args.prefills, payload["model"])

    messages = load_messages(args.messages)
    if args.stride > 1:
        messages = messages[:: args.stride]
    if args.max_cells:
        messages = messages[: args.max_cells]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    written = 0
    with args.out.open("w") as stream:
        for cell, record in enumerate(messages):
            for seed_index in SEEDS:
                for arm in ARMS if not args.arm else [args.arm]:
                    if prefills:
                        continuation = prefills[f"{record['level']}-{record['label']}-{seed_index}-{arm}"]
                    else:
                        continuation = continuation_for(model, tokenizer, record["text"], arm, seed_index, cell)
                    for probe_index, (emotion, probe_text) in enumerate(sorted(PROBES.items())):
                        transcript = build_transcript(
                            tokenizer,
                            record["text"],
                            continuation,
                            probe_text,
                            emotion,
                            roleplay=arm == "roleplay",
                            roleplay_user=ROLEPLAY_USER,
                            roleplay_assistant=ROLEPLAY_ASSISTANT,
                            assistant_word=markers["assistant_word"],
                        )
                        answer_seed = generation_seed(cell, seed_index, 2, probe_index, 0 if arm == "main" else 1)
                        generated = sample(model, tokenizer, transcript.input_ids, ANSWER_TOKENS, answer_seed)
                        position, answer = find_answer_position(tokenizer, generated, len(transcript.input_ids))
                        row = {
                            "level": record["level"],
                            "label": record["label"],
                            "seed": seed_index,
                            "arm": arm,
                            "probe": emotion,
                            "answer": answer,
                            "input_ids": transcript.input_ids
                            + (generated[: position - len(transcript.input_ids) + 1] if position >= 0 else []),
                            "span_labels": transcript.span_labels,
                            "emotion_position": transcript.emotion_position,
                            "assistant_position": transcript.assistant_position,
                            "answer_position": position,
                            "continuation": continuation,
                            "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
                        }
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        written += 1
            log.info(
                "cell %s/%s done, %s rows, %.1f min elapsed",
                cell + 1,
                len(messages),
                written,
                (time.monotonic() - started) / 60,
            )
    log.info("wrote %s transcripts to %s", written, args.out)


if __name__ == "__main__":
    main()
