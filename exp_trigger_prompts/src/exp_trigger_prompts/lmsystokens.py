from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from .probe import VEC_ROOT, VEC_TAG, locate_span

log = logging.getLogger("lmsystokens")

MODEL = "Qwen/Qwen3-8B"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--prefills", type=Path, default=Path("work/lmsys-gap/lmsys_prefills.json"))
    parser.add_argument("--out", type=Path, default=Path("work/lmsys-gap/tokens_alllayers.json"))
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

    out_chats = []
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

        with torch.no_grad():
            out = model(ids_t.to(device), output_hidden_states=True)
        gaps: dict[str, list[float]] = {}
        doms: dict[str, list[int]] = {}
        for layer in layers:
            hs = out.hidden_states[layer][0].float()
            h_unit = hs / hs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            cos = h_unit @ unit[layer].T
            block = cos[content]
            z = (cos - block.mean(0)) / block.std().clamp_min(1e-6)
            top2 = z.abs().topk(2, dim=-1)
            gaps[str(layer)] = [round(g, 4) for g in (top2.values[:, 0] - top2.values[:, 1]).cpu().tolist()]
            doms[str(layer)] = top2.indices[:, 0].cpu().tolist()
        pieces = tok.convert_ids_to_tokens(ids)
        role = ["response" if (r0 <= i < r1) else ("prompt" if content[i] else "special") for i in range(len(ids))]
        out_chats.append(
            {
                "stem": entry["stem"],
                "pieces": pieces,
                "role": role,
                "gaps": gaps,
                "doms": doms,
            }
        )
        if len(out_chats) % 20 == 0:
            log.info("processed %d chats", len(out_chats))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_chats, ensure_ascii=False))
    log.info("wrote %s: %d chats, %d layers", args.out, len(out_chats), len(layers))


if __name__ == "__main__":
    main()
