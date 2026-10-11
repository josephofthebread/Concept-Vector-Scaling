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

from exp_attn_interp.common import ValueCatcher, load_model
from exp_attn_interp.config import INSTRUCT_MODEL, vector_tag_for
from exp_attn_interp.decomp import MlpCatcher
from exp_attn_interp.gensurv import Hook
from exp_attn_interp.inject import last_pos_writes
from exp_attn_interp.vectors import load_diff_vectors

log = logging.getLogger("gensurv2")
TRIGGER = Path("../exp_trigger_prompts")
ROWS = (19, 22, 24, 28)


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--every", type=int, default=16)
    parser.add_argument("--prompts", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--t", type=int, default=15)
    parser.add_argument("--rows", default=",".join(map(str, ROWS)))
    parser.add_argument("--out", type=Path, default=Path("work/gensurv2"))
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
    catcher = ValueCatcher(model.model)
    mlps = MlpCatcher(model.model)
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
    hooks = {L: Hook(model, L) for L in rows}
    H = catcher.heads
    t = args.t
    started = time.monotonic()
    for n, k in enumerate(pairs):
        path = args.out / f"pair-{k:04d}.safetensors"
        if path.exists():
            continue
        attn = torch.zeros(len(rows), args.prompts, B, H)
        mlp = torch.zeros(len(rows), args.prompts, B)
        total = torch.zeros(len(rows), args.prompts)
        cur = torch.zeros(len(rows), args.prompts)
        v35 = vectors[last, k]
        for j in range(args.prompts):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": neutral[k][j]}], tokenize=False, add_generation_prompt=True, **kw
            )
            enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(dev)
            P = enc.input_ids.shape[1]
            for i, L in enumerate(rows):
                delta = vectors[L, k] * (args.alpha * norms[L])
                hk = hooks[L]
                hk.delta, hk.mode = delta, "last"
                gen = model.generate(
                    **enc,
                    max_new_tokens=t + 1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
                )
                hk.delta = None
                ids = gen[:, : P + t]
                if ids.shape[1] < P + t:
                    ids = torch.cat([ids, ids[:, -1:].expand(-1, P + t - ids.shape[1])], 1)
                o0 = model(input_ids=ids, output_attentions=True, output_hidden_states=True, use_cache=False)
                h0 = o0.hidden_states[last][0, -1].to(torch.float32)
                hw0, mw0 = last_pos_writes(model, catcher, mlps, o0, 0, last)
                hk.delta, hk.mode, hk.start = delta, "from_except_last", P - 1
                o1 = model(input_ids=ids, output_attentions=True, output_hidden_states=True, use_cache=False)
                h1 = o1.hidden_states[last][0, -1].to(torch.float32)
                hw1, mw1 = last_pos_writes(model, catcher, mlps, o1, 0, last)
                hk.mode = "from"
                o2 = model(input_ids=ids, output_hidden_states=True, use_cache=False)
                h2 = o2.hidden_states[last][0, -1].to(torch.float32)
                hk.delta = None
                dn = float(delta.norm())
                attn[i, j, :last] = ((hw1 - hw0) @ v35 / dn).cpu()
                mlp[i, j, :last] = ((mw1 - mw0) @ v35 / dn).cpu()
                total[i, j] = ((h1 - h0) @ v35 / dn).cpu()
                cur[i, j] = ((h2 - h0) @ v35 / dn).cpu()
                del o0, o1, o2
        save_file(
            {
                "attn": attn,
                "mlp": mlp,
                "total": total,
                "full": cur,
                "rows": torch.tensor(rows),
                "pair_id": torch.tensor(k),
                "t": torch.tensor(t),
            },
            str(path),
        )
        if n < 3 or n % 10 == 0:
            log.info(
                "%d/%d pair %d: context-only survival by row %s (full %s); attention part %s, MLP part %s; recon %s; %.1f min",
                n,
                len(pairs),
                k,
                [round(float(total[i].mean()), 2) for i in range(len(rows))],
                [round(float(cur[i].mean()), 2) for i in range(len(rows))],
                [round(float(attn[i].sum((1, 2)).mean()), 2) for i in range(len(rows))],
                [round(float(mlp[i].sum(1).mean()), 2) for i in range(len(rows))],
                [
                    round(float((attn[i].sum((1, 2)) + mlp[i].sum(1) - total[i]).abs().mean()), 3)
                    for i in range(len(rows))
                ],
                (time.monotonic() - started) / 60,
            )
    for hk in hooks.values():
        hk.remove()
    log.info("done %d pairs, %.1f min", len(pairs), (time.monotonic() - started) / 60)


if __name__ == "__main__":
    main()
