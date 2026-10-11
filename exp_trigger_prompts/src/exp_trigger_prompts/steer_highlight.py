from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from .probe import VEC_ROOT, VEC_TAG, locate_span, render_svg

log = logging.getLogger("steer-highlight")

MODEL = "Qwen/Qwen3-8B"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--demo", type=Path, default=Path("data/steer_demo.json"))
    parser.add_argument("--out-root", type=Path, default=Path("work/STEER_DEMO"))
    args = parser.parse_args()

    demo = json.loads(args.demo.read_text())

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
    )
    model.eval()
    device = model.device
    with safe_open(VEC_ROOT / VEC_TAG[MODEL] / "diff.safetensors", framework="pt") as handle:
        diff = handle.get_tensor("diff")

    for idx, e in enumerate(demo, 1):
        pid, hl = e["pair"], e["hl_layer"]
        out_dir = args.out_root / f"{idx}-{pid:04d}"

        def emit(key: str, reply: str) -> int:
            prompt = e["prompt"]
            direction = diff[hl, pid].float().to(device)
            direction = direction / direction.norm().clamp_min(1e-8)
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]
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
            ids = cast("list[int]", ids_t[0].tolist())
            with torch.no_grad():
                out = model(ids_t.to(device), output_hidden_states=True)
            hidden = out.hidden_states[hl][0].float()
            hidden = hidden / hidden.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            cos = hidden @ direction

            p0, p1 = locate_span(ids, prompt, tok)
            r0, r1 = locate_span(ids, reply, tok, start=max(p1, 0))
            content = torch.zeros(len(ids), dtype=torch.bool)
            if p0 >= 0:
                content[p0:p1] = True
            if r0 >= 0:
                content[r0:r1] = True
            if content.sum() < 2 or r0 < 0:
                content[:] = True
                r0, r1 = (r0 if r0 >= 0 else 0), (r1 if r1 > 0 else len(ids))
            z = (cos - cos[content].mean()) / cos[content].std().clamp_min(1e-6)

            display = [cast("str", tok.decode([i])) for i in ids]
            toks, zv = display[r0:r1], z[r0:r1].tolist()
            render_svg(toks, zv, 0, "", [], out_dir / f"{key}.svg")
            return len(toks)

        n = emit("baseline", e["baseline"])
        log.info("concept %d pair %d [%s|%s] hl_L%d: baseline %d tokens", idx, pid, e["pos_pole"], e["neg_pole"], hl, n)
        for arm in e["arms"]:
            n = emit(arm["key"], arm["text"])
            log.info(
                "  %s  L%d a%+.2f%s -> %d tokens",
                arm["key"],
                arm["layer"],
                arm["alpha"],
                "  [nonsense]" if arm.get("nonsense") else "",
                n,
            )

    log.info("done: %d concepts -> %s", len(demo), args.out_root)


if __name__ == "__main__":
    main()
