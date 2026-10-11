from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

from exp_attn_interp.common import load_model
from exp_attn_interp.config import INSTRUCT_MODEL, vector_tag_for
from exp_attn_interp.vectors import load_diff_vectors

log = logging.getLogger("basepos")
TRIGGER = Path("../exp_trigger_prompts")
ROWS = (19, 22, 35)


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("work/basepos.npz"))
    parser.add_argument("--rows", default="19,22,35")
    parser.add_argument("--sweep", default="sweep-Qwen_Qwen3-8B.jsonl")
    parser.add_argument(
        "--storyproj",
        type=Path,
        default=Path("work/storyproj"),
        help="pole-gap normalization; if missing, raw projections are stored",
    )
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = tuple(int(x) for x in args.rows.split(","))
    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    vectors = load_diff_vectors(device=str(model.device), tag=vector_tag_for(args.model))
    dev = model.device
    kw: dict[str, Any] = {"enable_thinking": False} if "Qwen3" in args.model else {}
    base: dict[int, dict[int, tuple[str, str]]] = {}
    with (TRIGGER / "work" / "steer" / args.sweep).open() as fh:
        for line in fh:
            if ':baseline"' not in line:
                continue
            d = json.loads(line)
            if d["alpha"] == 0.0:
                base.setdefault(int(d["pair_id"]), {})[int(d["slot"])] = (d["prompt"], d["response"])
    pairs = sorted(base)[:: args.every]
    gap = {}
    for k in pairs:
        f = args.storyproj / f"pair-{k:04d}.safetensors"
        if f.exists():
            with safe_open(str(f), framework="np") as h:
                pooled = h.get_tensor("pooled")
            gap[k] = pooled[0].mean(0) - pooled[1].mean(0), 0.5 * (pooled[0].mean(0) + pooled[1].mean(0))
    pos = np.full((len(pairs), args.slots, len(rows)), np.nan, dtype=np.float32)
    for n, k in enumerate(pairs):
        slots = sorted(base[k])[: args.slots]
        for j, s in enumerate(slots):
            prompt, resp = base[k][s]
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, **kw
            )
            ids_p = tokenizer(text, add_special_tokens=False)["input_ids"]
            ids_r = tokenizer(resp, add_special_tokens=False)["input_ids"][:256]
            if len(ids_r) < 4:
                continue
            ids = torch.tensor([ids_p + ids_r], device=dev)
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
            P = len(ids_p)
            for r, L in enumerate(rows):
                h = out.hidden_states[L][0, P:].to(torch.float32).mean(0)
                proj = float(h @ vectors[L, k])
                if k in gap:
                    g, mid = gap[k]
                    pos[n, j, r] = (proj - float(mid[L])) / (0.5 * float(g[L]) + 1e-6)
                else:
                    pos[n, j, r] = proj
        if n % 100 == 0:
            log.info(
                "%d/%d pair %d: baseline position at rows %s: %s",
                n,
                len(pairs),
                k,
                rows,
                [round(float(np.nanmean(pos[n, :, r])), 2) for r in range(len(rows))],
            )
    np.savez(str(args.out), pos=pos, pairs=np.array(pairs), rows=np.array(rows))
    log.info(
        "done; median position at the middle row: %.2f (p10 %.2f p90 %.2f)",
        np.nanmedian(pos[:, :, 1]),
        np.nanpercentile(pos[:, :, 1], 10),
        np.nanpercentile(pos[:, :, 1], 90),
    )


if __name__ == "__main__":
    main()
