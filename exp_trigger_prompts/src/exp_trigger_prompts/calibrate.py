from __future__ import annotations

import json
import logging
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch

from . import steering

log = logging.getLogger("trigger-calibrate")


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--baselines", type=Path, required=True, help="baselines-<model>.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="norms JSON destination")
    parser.add_argument("--sample", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    layers = steering.LAYERS[args.model]
    rows = [json.loads(line) for line in args.baselines.open() if line.strip()]
    rows = [r for r in rows if r.get("kind") == "neutral" and r.get("response")]
    random.Random(0).shuffle(rows)
    rows = rows[: args.sample]
    log.info("%s: calibrating norms at layers %s over %d neutral baselines", args.model, layers, len(rows))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, steering.DTYPE),
        device_map="cuda",
        attn_implementation=steering.attn_impl(args.model),
    ).eval()

    template_kwargs: dict[str, Any] = {"enable_thinking": False} if "Qwen3" in args.model else {}
    sums = {layer: 0.0 for layer in layers}
    counts = {layer: 0 for layer in layers}

    captured: dict[int, torch.Tensor] = {}

    def hook(layer: int):
        def fn(_module: Any, _inputs: Any, output: Any) -> None:
            captured[layer] = (output[0] if isinstance(output, tuple) else output).detach()

        return fn

    handles = [model.model.layers[layer - 1].register_forward_hook(hook(layer)) for layer in layers]
    try:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            prefixes: list[str] = []
            fulls: list[str] = []
            for r in batch:
                messages = [{"role": "user", "content": r["prompt"]}]
                prefix = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **template_kwargs
                )
                assert isinstance(prefix, str)
                prefixes.append(prefix)
                fulls.append(prefix + r["response"])
            enc = tokenizer(fulls, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            prefix_lens = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prefixes]
            full_lens = enc.attention_mask.sum(dim=1).tolist()
            model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
            seq = enc.input_ids.shape[1]
            for layer in layers:
                hidden = captured[layer].float()
                for row, (plen, flen) in enumerate(zip(prefix_lens, full_lens)):
                    resp = int(flen) - int(plen)
                    if resp <= 0:
                        continue
                    norms = hidden[row, seq - resp : seq, :].norm(dim=-1)
                    sums[layer] += float(norms.sum())
                    counts[layer] += int(resp)
            captured.clear()
    finally:
        for handle in handles:
            handle.remove()

    norms = {str(layer): (sums[layer] / counts[layer]) for layer in layers if counts[layer]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": args.model, "layers": layers, "norms": norms}, indent=2))
    log.info("wrote %s: %s", args.out, {k: round(v, 1) for k, v in norms.items()})


if __name__ == "__main__":
    main()
