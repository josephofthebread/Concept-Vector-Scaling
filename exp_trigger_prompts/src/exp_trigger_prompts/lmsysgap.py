from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from .probe import VEC_ROOT, VEC_TAG, locate_span

log = logging.getLogger("lmsysgap")

MODEL = "Qwen/Qwen3-8B"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--prefills", type=Path, default=Path("work/lmsys-gap/lmsys_prefills.json"))
    parser.add_argument("--out", type=Path, default=Path("work/lmsys-gap/points.npz"))
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    prefills = json.loads(args.prefills.read_text())
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
    )
    model.eval()
    device = model.device
    n_layers = model.config.num_hidden_layers
    layers = list(range(n_layers + 1))
    with safe_open(VEC_ROOT / VEC_TAG[MODEL] / "diff.safetensors", framework="pt") as handle:
        diff = handle.get_tensor("diff").float().to(device)
    unit = diff / diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    layer_col: list[np.ndarray] = []
    gap_col: list[np.ndarray] = []
    kept = 0
    for entry in prefills:
        messages = entry["conversation"]
        enc = cast(
            "Any",
            tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            ),
        )  # type: ignore[arg-type]
        ids_t = enc["input_ids"]
        if ids_t.shape[1] > args.max_tokens:
            continue
        ids = cast("list[int]", ids_t[0].tolist())
        p0, p1 = locate_span(ids, messages[0]["content"], tok)
        r0, r1 = locate_span(ids, messages[1]["content"], tok, start=max(p1, 0))
        if r0 < 0 or r1 - r0 < 2:
            continue
        content = torch.zeros(len(ids), dtype=torch.bool)
        if p0 >= 0:
            content[p0:p1] = True
        content[r0:r1] = True
        resp = list(range(r0, r1))

        with torch.no_grad():
            out = model(ids_t.to(device), output_hidden_states=True)
        for layer in layers:
            hidden = out.hidden_states[layer][0].float()
            h_unit = hidden / hidden.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            cos = h_unit @ unit[layer].T
            block = cos[content]
            z = (cos - block.mean(0)) / block.std().clamp_min(1e-6)
            top2 = z[resp].abs().topk(2, dim=-1).values
            gap = (top2[:, 0] - top2[:, 1]).cpu().numpy().astype(np.float32)
            layer_col.append(np.full(gap.shape, layer, dtype=np.int8))
            gap_col.append(gap)
        kept += 1
        if kept % 10 == 0:
            log.info("processed %d/%d chats", kept, len(prefills))

    layer_arr = np.concatenate(layer_col)
    gap_arr = np.concatenate(gap_col)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, layer=layer_arr, gap=gap_arr, n_chats=np.int32(kept))
    log.info("wrote %s: %d chats, %d points", args.out, kept, gap_arr.size)


if __name__ == "__main__":
    main()
