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

log = logging.getLogger("margintok")

MODEL = "Qwen/Qwen3-8B"
LAYERS = [18, 21, 24]
MAX_TOKENS = 10000


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--prefills", type=Path, default=Path("work/margin/prefills.json"))
    parser.add_argument("--out", type=Path, default=Path("work/margin/margin.json"))
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    prefills = json.loads(args.prefills.read_text())
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
    )
    model.eval()
    device = model.device
    with safe_open(VEC_ROOT / VEC_TAG[MODEL] / "diff.safetensors", framework="pt") as handle:
        diff = handle.get_tensor("diff").float().to(device)
    unit = {L: (diff[L] / diff[L].norm(dim=-1, keepdim=True).clamp_min(1e-8)) for L in LAYERS}

    margins: dict[int, list[np.ndarray]] = {L: [] for L in LAYERS}
    total = 0
    for entry in prefills:
        if total >= MAX_TOKENS:
            break
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
        content = torch.zeros(len(ids), dtype=torch.bool)
        if p0 >= 0:
            content[p0:p1] = True
        if r0 >= 0:
            content[r0:r1] = True
        cidx = content.nonzero(as_tuple=True)[0]
        if cidx.numel() < 2:
            continue

        with torch.no_grad():
            out = model(ids_t.to(device), output_hidden_states=True)
        for L in LAYERS:
            hs = out.hidden_states[L][0].float()
            h_unit = hs / hs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            cos = h_unit @ unit[L].T
            block = cos[content]
            z = (cos - block.mean(0)) / block.std().clamp_min(1e-6)
            a = z.abs()[cidx]
            top2 = a.topk(2, dim=-1)
            m = a - top2.values[:, 0:1]
            arg = top2.indices[:, 0]
            m[torch.arange(a.shape[0]), arg] = top2.values[:, 0] - top2.values[:, 1]
            margins[L].append(m.cpu().numpy().astype(np.float32))
        total += int(cidx.numel())
        if total % 2000 < 200:
            log.info("%d content tokens", total)

    result: dict[str, Any] = {"layers": LAYERS, "n_tokens": total}
    for L in LAYERS:
        M = np.concatenate(margins[L], 0)
        result[str(L)] = np.percentile(M, 95, axis=0).round(4).tolist()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False))
    log.info("wrote %s (%d tokens)", args.out, total)


if __name__ == "__main__":
    main()
