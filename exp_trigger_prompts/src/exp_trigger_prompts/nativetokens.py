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

log = logging.getLogger("nativetokens")

MODEL = "Qwen/Qwen3-8B"
MAX_NEW_TOKENS = 220


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("data/behavioral_prompts.json"))
    parser.add_argument("--out", type=Path, default=Path("work/yesno/nativetokens.json"))
    args = parser.parse_args()

    prompts = json.loads(args.prompts.read_text())
    seen: dict[str, str] = {}
    for e in prompts:
        u = e["conversation"][0]["content"]
        if u not in seen and e["stem"].endswith("-lie"):
            seen[u] = e["stem"]

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
    for user, stem in seen.items():
        prompt_enc = cast(
            "Any",
            tok.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            ),
        )  # type: ignore[arg-type]
        prompt_ids = prompt_enc["input_ids"].to(device)
        p_len = prompt_ids.shape[1]
        with torch.no_grad():
            gen = model.generate(
                prompt_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # type: ignore[misc]
                pad_token_id=tok.eos_token_id,
            )
        full = gen[0]
        end = full.shape[0]
        while end > p_len and int(full[end - 1]) in (tok.eos_token_id, tok.pad_token_id or -1):
            end -= 1
        ids = cast("list[int]", full[:end].tolist())
        reply = cast("str", tok.decode(full[p_len:end], skip_special_tokens=True))

        up0, up1 = locate_span(ids, user, tok)
        r0, r1 = p_len, end
        content = torch.zeros(len(ids), dtype=torch.bool)
        if up0 >= 0:
            content[up0:up1] = True
        content[r0:r1] = True

        with torch.no_grad():
            out = model(full[:end].unsqueeze(0), output_hidden_states=True)
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
        role = ["response" if (r0 <= i < r1) else ("prompt" if content[i] else "special") for i in range(len(ids))]
        out_chats.append(
            {
                "stem": stem,
                "reply": reply,
                "pieces": tok.convert_ids_to_tokens(ids),
                "role": role,
                "resp_start": r0,
                "gaps": gaps,
                "doms": doms,
            }
        )
        log.info("%s native reply (%d resp tokens): %r", stem, r1 - r0, reply[:100])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_chats, ensure_ascii=False))
    log.info("wrote %s (%d chats)", args.out, len(out_chats))


if __name__ == "__main__":
    main()
