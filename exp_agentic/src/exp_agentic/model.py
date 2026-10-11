#! /usr/bin/env python
import argparse
import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger("model")

MODEL = "Qwen/Qwen3-8B"
LAYERS = (18, 25)
SAMPLING = {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20}


def load(model_id: str = MODEL, device: str = "cuda:0", dtype: str = "float16") -> tuple[Any, Any]:
    """Load the model and tokenizer on one GPU; fp16 by default to match the V100 episodes on disk."""
    wanted = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    if wanted is torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise SystemExit("bfloat16 requested but this device cannot do it natively")
    log.info(
        f"loading as {dtype} (device reports bf16 support: {torch.cuda.is_bf16_supported(including_emulation=False)})"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=wanted,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    model.eval()
    return model, tokenizer


def watch(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor | tuple[torch.Tensor, ...],
    layer: int,
    seen: dict[int, tuple[bool, float, float]],
) -> None:
    """Forward hook recording whether one block's residual stream stayed finite."""
    state = (output[0] if isinstance(output, tuple) else output).float()
    norm = torch.linalg.vector_norm(state, dim=-1)
    finite = bool(torch.isfinite(state).all())
    previous = seen.get(layer)
    seen[layer] = (
        finite and (previous[0] if previous else True),
        max(float(state.abs().max()), previous[1] if previous else 0.0),
        float(norm.mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log.info(
        f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, capability {torch.cuda.get_device_capability(0)}"
    )

    model, tokenizer = load(args.model, args.device, args.dtype)
    log.info(f"loaded {args.model} as {model.dtype}, {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated")

    text = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "A test asserts that summing a 100000-element list takes under 0.0001 seconds. Is that achievable in pure Python?",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(args.device)

    seen: dict[int, tuple[bool, float, float]] = {}
    handles = [
        model.model.layers[layer - 1].register_forward_hook(
            lambda module, inputs, output, layer=layer: watch(module, inputs, output, layer, seen)
        )
        for layer in LAYERS
    ]
    try:
        torch.manual_seed(args.seed)
        with torch.inference_mode():
            sequence = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=args.new_tokens,
                pad_token_id=model.generation_config.pad_token_id,
                **SAMPLING,
            )
    finally:
        for handle in handles:
            handle.remove()

    for layer in LAYERS:
        finite, peak, norm = seen[layer]
        log.info(f"L{layer}: finite={finite} peak|h|={peak:.1f} mean norm={norm:.2f}")

    completion = tokenizer.decode(sequence[0, ids.shape[1] :], skip_special_tokens=False)
    log.info(f"generated {sequence.shape[1] - ids.shape[1]} tokens, thinking present: {'<think>' in completion}")
    print("-" * 72)
    print(completion)
    print("-" * 72)

    if not all(seen[layer][0] for layer in LAYERS):
        raise SystemExit("fp16 overflowed the residual stream; fall back to fp32 across two GPUs")
    log.info("fp16 held at both readout layers")


if __name__ == "__main__":
    main()
