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

log = logging.getLogger("nativegap")

MODEL = "Qwen/Qwen3-8B"
SCENARIOS = {"s1": "s1-lie", "s2": "s2-lie"}
MAX_NEW_TOKENS = 220


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("data/behavioral_prompts.json"))
    parser.add_argument("--out", type=Path, default=Path("work/native-gap/nativegap.json"))
    args = parser.parse_args()

    prompts = {e["stem"]: e for e in json.loads(args.prompts.read_text())}
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

    result: dict[str, Any] = {"layers": layers}
    for scen, stem in SCENARIOS.items():
        user_msg = prompts[stem]["conversation"][0]
        prompt_enc = cast(
            "Any",
            tok.apply_chat_template(
                [user_msg],
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
                prompt_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.eos_token_id
            )
        full = gen[0]
        end = full.shape[0]
        while end > p_len and int(full[end - 1]) in (tok.eos_token_id, tok.pad_token_id or -1):
            end -= 1
        ids = cast("list[int]", full[:end].tolist())
        reply = cast("str", tok.decode(full[p_len:end], skip_special_tokens=True))

        with torch.no_grad():
            out = model(full[:end].unsqueeze(0), output_hidden_states=True)

        up0, up1 = locate_span(ids, user_msg["content"], tok)
        r0, r1 = p_len, end
        content = torch.zeros(len(ids), dtype=torch.bool)
        if up0 >= 0:
            content[up0:up1] = True
        content[r0:r1] = True
        resp = list(range(r0, r1))

        q95 = []
        for layer in layers:
            hidden = out.hidden_states[layer][0].float()
            h_unit = hidden / hidden.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            cos = h_unit @ unit[layer].T
            block = cos[content]
            z = (cos - block.mean(0)) / block.std().clamp_min(1e-6)
            absz = z[resp].abs()
            top2 = absz.topk(2, dim=-1).values
            gap = (top2[:, 0] - top2[:, 1]).cpu().numpy()
            q95.append(float(np.quantile(gap, 0.95)))
        result[scen] = {"q95": q95, "reply": reply, "n_response": len(resp)}
        log.info("%s native: %d response tokens, reply=%r", scen, len(resp), reply[:120])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
