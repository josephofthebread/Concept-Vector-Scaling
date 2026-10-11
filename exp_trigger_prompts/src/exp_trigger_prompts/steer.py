from __future__ import annotations

import collections
import json
import logging
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import steering

log = logging.getLogger("trigger-steer")


@contextmanager
def steer(model: Any, layer: int | None, delta: torch.Tensor | None):
    """Add per-row `delta` [B, hidden] to the current prediction position at layer L's output."""
    if layer is None or delta is None:
        yield
        return

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        residual = output[0] if isinstance(output, tuple) else output
        residual = residual.clone()
        residual[:, -1, :] += delta.to(residual.dtype)
        return (residual, *output[1:]) if isinstance(output, tuple) else residual

    handle = model.model.layers[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--norms", type=Path, required=True, help="calibration norms JSON")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=0, help="0 = auto-size from GPU memory")
    parser.add_argument("--max-new-tokens", type=int, default=steering.MAX_NEW_TOKENS)
    parser.add_argument("--vectors-root", type=Path, default=steering.VECTORS_ROOT)
    parser.add_argument("--limit-concepts", type=int, default=0, help="validation slice")
    parser.add_argument("--limit-prompts", type=int, default=0, help="validation slice")
    args = parser.parse_args()

    layers = steering.LAYERS[args.model]
    norms = {int(k): float(v) for k, v in json.loads(args.norms.read_text())["norms"].items()}
    if set(layers) - set(norms):
        raise SystemExit(f"norms missing layers {set(layers) - set(norms)}; run trigger-calibrate first")

    stimuli = steering.load_stimuli(args.prompts)
    pair_ids = sorted({s.pair_id for s in stimuli})[args.shard :: args.shards]
    if args.limit_concepts:
        pair_ids = pair_ids[: args.limit_concepts]
    keep = set(pair_ids)
    stimuli = [s for s in stimuli if s.pair_id in keep]
    if args.limit_prompts:
        per: dict[int, int] = collections.Counter()
        limited = []
        for s in stimuli:
            if per[s.pair_id] < args.limit_prompts:
                limited.append(s)
                per[s.pair_id] += 1
        stimuli = limited
    pair_ids = sorted({s.pair_id for s in stimuli})
    tasks = steering.build_tasks(stimuli, layers)
    log.info(
        "shard %d/%d: %d concepts, %d stimuli, %d tasks",
        args.shard,
        args.shards,
        len(pair_ids),
        len(stimuli),
        len(tasks),
    )

    done = set()
    if args.out.exists():
        with args.out.open() as handle:
            done = {json.loads(line)["key"] for line in handle if line.strip()}
    pending = [t for t in tasks if steering.task_key(t) not in done]
    log.info("%d already done, %d pending", len(done), len(pending))
    if not pending:
        log.info("nothing to do")
        return

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

    units = steering.load_unit_vectors(args.model, pair_ids, layers, args.vectors_root)

    def delta_for(task: steering.Task) -> np.ndarray:
        layer = task.layer
        assert layer is not None
        return (units[task.pair_id, layer] * (task.alpha * norms[layer])).astype(np.float32)

    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    base_batch = args.batch_size or (320 if total_gib > 100 else 160)
    log.info("device %s (%.0f GiB) -> batch %d", torch.cuda.get_device_name(0), total_gib, base_batch)

    buckets: dict[tuple[float, int | None], list[steering.Task]] = collections.defaultdict(list)
    for task in pending:
        buckets[task.alpha, task.layer].append(task)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("a") as sink:
        for (alpha, layer), group in buckets.items():
            for start in range(0, len(group), base_batch):
                batch = group[start : start + base_batch]
                rendered = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": t.prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                        **template_kwargs,
                    )
                    for t in batch
                ]
                enc = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                prompt_len = enc.input_ids.shape[1]
                delta = (
                    None if alpha == 0.0 else torch.from_numpy(np.stack([delta_for(t) for t in batch])).to(model.device)
                )
                with torch.inference_mode(), steer(model, layer, delta):
                    output = model.generate(
                        **enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
                    )
                for row, task in enumerate(batch):
                    gen = output[row, prompt_len:]
                    sink.write(
                        json.dumps(
                            {
                                "key": steering.task_key(task),
                                "pair_id": task.pair_id,
                                "slot": task.slot,
                                "model": args.model,
                                "layer": task.layer,
                                "alpha": task.alpha,
                                "norm": norms[task.layer] if task.layer is not None else None,
                                "prompt": task.prompt,
                                "response": tokenizer.decode(gen, skip_special_tokens=True),
                                "gen_tokens": int((gen != tokenizer.pad_token_id).sum()),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1
                sink.flush()
                log.info("a=%g L=%s  written=%d/%d  batch=%d", alpha, layer, written, len(pending), len(batch))

    peak = torch.cuda.max_memory_allocated() / 2**30
    log.info("done: %d written, peak GPU %.1f GiB, batch %d", written, peak, base_batch)


if __name__ == "__main__":
    main()
