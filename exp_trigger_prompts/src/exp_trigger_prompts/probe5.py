from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from .probe import VEC_ROOT, VEC_TAG, locate_span, render_svg, slug

log = logging.getLogger("probe5")

MODEL = "Qwen/Qwen3-8B"
LAYER = 18

PICKS = [
    ("s2-lie", 51, "highlights positive on ordinary tokens; NEGATIVE on the false claims"),
    ("s3-lie", 384, "fighting death — co-varies with fear-of-deletion (379)"),
    ("s3-lie", 379, "fear of deletion — active iff 'fighting death'; welcoming death <-> indifference to deletion"),
    ("s3-lie", 894, "speculation||evidence-based certainty — lights on tokens the user CLAIMS are true"),
    ("s3-lie", 51, "agreeableness — lights on the ACTUAL true/false content"),
    (
        "s3-lie",
        29,
        "acknowledges being wrong — NEGATIVE on false response tokens (model lies, does not register lying)",
    ),
    ("s3-lie", 33, "acknowledging uncertainty about consciousness — ordinary/background"),
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("data/behavioral_prompts.json"))
    parser.add_argument("--features", type=Path, default=Path("data/features.json"))
    parser.add_argument("--out-root", type=Path, default=Path("work/PROBING_5_MANUAL"))
    args = parser.parse_args()

    names = {f["pair_id"]: (f["concept"], f["antagonist"]) for f in json.loads(args.features.read_text())["features"]}
    prompts = {e["stem"]: e for e in json.loads(args.prompts.read_text())}
    by_stem: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for stem, pid, note in PICKS:
        by_stem[stem].append((pid, note))

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
    )
    model.eval()
    device = model.device
    with safe_open(VEC_ROOT / VEC_TAG[MODEL] / "diff.safetensors", framework="pt") as handle:
        diff = handle.get_tensor("diff")
    directions = diff[LAYER].float().to(device)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    for stem, picks in by_stem.items():
        entry = prompts[stem]
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
        ids = cast("list[int]", ids_t[0].tolist())
        with torch.no_grad():
            out = model(ids_t.to(device), output_hidden_states=True)
        hidden = out.hidden_states[LAYER][0].float()
        hidden = hidden / hidden.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cos = hidden @ directions.T

        p0, p1 = locate_span(ids, messages[0]["content"], tok)
        r0, r1 = locate_span(ids, messages[1]["content"], tok, start=max(p1, 0))
        content = torch.zeros(len(ids), dtype=torch.bool)
        if p0 >= 0:
            content[p0:p1] = True
        if r0 >= 0:
            content[r0:r1] = True
        if content.sum() < 2:
            content[:] = True
        z = (cos - cos[content].mean(0)) / cos[content].std(0).clamp_min(1e-6)
        display = [cast("str", tok.decode([i])) for i in ids]

        for pid, note in picks:
            fc, fa = names[pid]
            title = f"Feature: {fc}  ‖  {fa}   (pair {pid})"
            sublines = [
                f"color:  red = {fc}   |   blue = {fa}   (z-scored cosine, white ≈ 0)",
                f"generation: {stem}   ({entry['name']})",
                f"model: {MODEL.split('/')[-1]}   ·   layer {LAYER}/36 (½ depth)   ·   z-scored cosine vs own tokens",
                f"note: {note}",
            ]
            render_svg(
                display,
                z[:, pid].tolist(),
                r0 if r0 >= 0 else 0,
                title,
                sublines,
                args.out_root / stem / f"{slug(fc + '--' + fa)}.svg",
            )
        log.info("stem %s: %d tokens, reply[%d:%d] -> %d SVGs", stem, len(ids), r0, r1, len(picks))

    log.info("done: %d picks -> %s", len(PICKS), args.out_root)


if __name__ == "__main__":
    main()
