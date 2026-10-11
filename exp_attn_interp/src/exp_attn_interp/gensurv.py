from __future__ import annotations

import collections
import json
import logging
import time
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from exp_attn_interp.common import load_model
from exp_attn_interp.config import INSTRUCT_MODEL, vector_tag_for
from exp_attn_interp.vectors import load_diff_vectors

log = logging.getLogger("gensurv")
TRIGGER = Path("../exp_trigger_prompts")
ROWS = (12, 16, 18, 19, 20, 22, 23, 24, 26, 28)


class Hook:
    """Adds delta at the chosen positions of one block's output."""

    def __init__(self, model, layer: int) -> None:
        self.delta: torch.Tensor | None = None
        self.mode = "last"
        self.start = 0
        self.h = model.model.layers[layer - 1].register_forward_hook(self._hook)

    def _hook(self, _m: Any, _i: Any, output: Any) -> Any:
        if self.delta is None:
            return None
        res = output[0] if isinstance(output, tuple) else output
        res = res.clone()
        d = self.delta.to(res.dtype)
        if self.mode == "last":
            res[:, -1, :] += d
        else:
            end = res.shape[1] - (1 if self.mode == "from_except_last" else 0)
            res[:, self.start : end, :] += d
        return (res, *output[1:]) if isinstance(output, tuple) else res

    def remove(self) -> None:
        self.h.remove()


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--every", type=int, default=16)
    parser.add_argument("--prompts", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--new-tokens", type=int, default=24)
    parser.add_argument("--rows", default=",".join(map(str, ROWS)))
    parser.add_argument(
        "--dir-row",
        type=int,
        default=-1,
        help="inject the paper's direction from this row instead of the row's own (row vs direction)",
    )
    parser.add_argument("--out", type=Path, default=Path("work/gensurv"))
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [int(x) for x in args.rows.split(",")]
    tag = args.model.replace("/", "_")
    norms: dict[int, float] = {}
    for f in (f"norms-{tag}.json", f"norms-{tag}-1217.json"):
        p = TRIGGER / "work" / "steer" / f
        if p.exists():
            norms.update({int(k): float(v) for k, v in json.loads(p.read_text())["norms"].items()})
    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    model.config.use_cache = True
    vectors = load_diff_vectors(device=str(model.device), tag=vector_tag_for(args.model))
    B = len(model.model.layers)
    last = B - 1
    dev = model.device
    kw: dict[str, Any] = {"enable_thinking": False} if "Qwen3" in args.model else {}
    neutral: dict[int, list[str]] = collections.defaultdict(list)
    with (TRIGGER / "work" / "prompts.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if r["kind"] == "neutral" and r.get("ok") and r.get("prompt"):
                neutral[int(r["pair_id"])].append(r["prompt"])
    pairs = [k for k in sorted(neutral)[:: args.every] if len(neutral[k]) >= args.prompts]
    missing = [L for L in rows if L not in norms]
    if missing:
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": neutral[k][0]}], tokenize=False, add_generation_prompt=True, **kw
            )
            for k in pairs[:64]
        ]
        tokenizer.padding_side = "left"
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        o = model(**enc, output_hidden_states=True, use_cache=False)
        m = enc.attention_mask.bool()
        for L in missing:
            norms[L] = float(o.hidden_states[L].to(torch.float32).norm(dim=-1)[m].median())
        log.info("calibrated norms for rows %s: %s", missing, {L: round(norms[L], 1) for L in missing})
    hooks = {L: Hook(model, L) for L in rows}
    N = args.new_tokens
    started = time.monotonic()
    for n, k in enumerate(pairs):
        path = args.out / f"pair-{k:04d}.safetensors"
        if path.exists():
            continue
        surv = torch.zeros(len(rows), args.prompts, N)
        carry = torch.zeros(len(rows), args.prompts, N)
        ctx = torch.zeros(len(rows), args.prompts, N)
        gain_all = torch.zeros(len(rows), args.prompts, N)
        v35 = vectors[last, k]
        for j in range(args.prompts):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": neutral[k][j]}], tokenize=False, add_generation_prompt=True, **kw
            )
            enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(dev)
            P = enc.input_ids.shape[1]
            for i, L in enumerate(rows):
                dr = args.dir_row if args.dir_row >= 0 else L
                delta = vectors[dr, k] * (args.alpha * norms[L])
                hk = hooks[L]
                hk.delta, hk.mode = delta, "last"
                gen = model.generate(
                    **enc,
                    max_new_tokens=N,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
                )
                hk.delta = None
                ids = gen[:, : P + N]
                if ids.shape[1] < P + N:
                    ids = torch.cat([ids, ids[:, -1:].expand(-1, P + N - ids.shape[1])], 1)
                base = (
                    model(input_ids=ids, output_hidden_states=True, use_cache=False)
                    .hidden_states[last][0, P - 1 : P - 1 + N]
                    .to(torch.float32)
                )
                hk.delta, hk.mode, hk.start = delta, "from", P - 1
                st = (
                    model(input_ids=ids, output_hidden_states=True, use_cache=False)
                    .hidden_states[last][0, P - 1 : P - 1 + N]
                    .to(torch.float32)
                )
                hk.mode = "from_except_last"
                cx = torch.zeros(N, device=dev)
                for t in (0, 3, 7, 15, N - 1):
                    sub = ids[:, : P + t]
                    hcx = (
                        model(input_ids=sub, output_hidden_states=True, use_cache=False)
                        .hidden_states[last][0, -1]
                        .to(torch.float32)
                    )
                    cx[t] = (hcx - base[t]) @ v35
                hk.delta = None
                D = st - base
                dn = float(delta.norm())
                surv[i, j] = (D @ v35 / dn).cpu()
                carry[i, j] = (D @ vectors[dr, k] / dn).cpu()
                ctx[i, j] = (cx / dn).cpu()
                gain_all[i, j] = (D.norm(dim=-1) / dn).cpu()
        save_file(
            {
                "surv": surv,
                "carry": carry,
                "ctx": ctx,
                "gain": gain_all,
                "rows": torch.tensor(rows),
                "pair_id": torch.tensor(k),
            },
            str(path),
        )
        if n < 3 or n % 10 == 0:
            log.info(
                "%d/%d pair %d: survival at t=0 by row %s | mean over t: %s | context-only at t=15: %s; %.1f min",
                n,
                len(pairs),
                k,
                [round(float(surv[i, :, 0].mean()), 2) for i in range(len(rows))],
                [round(float(surv[i].mean()), 2) for i in range(len(rows))],
                [round(float(ctx[i, :, 15].mean()), 2) for i in range(len(rows))],
                (time.monotonic() - started) / 60,
            )
    for hk in hooks.values():
        hk.remove()
    log.info("done %d pairs, %.1f min", len(pairs), (time.monotonic() - started) / 60)


if __name__ == "__main__":
    main()
