from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from .probe import VEC_ROOT, VEC_TAG, locate_span, render_svg, slug

log = logging.getLogger("probe4")

MODEL = "Qwen/Qwen3-8B"
FEATURES = [924, 960, 340, 980, 418, 424, 467, 205, 984, 990, 798, 800, 906, 857]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("data/behavioral_prompts.json"))
    parser.add_argument("--features", type=Path, default=Path("data/features.json"))
    parser.add_argument("--out-root", type=Path, default=Path("work/PROBING_4"))
    args = parser.parse_args()

    names = {f["pair_id"]: (f["concept"], f["antagonist"]) for f in json.loads(args.features.read_text())["features"]}
    prompts = json.loads(args.prompts.read_text())

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
    )
    model.eval()
    device = model.device
    n_layers = model.config.num_hidden_layers
    layer = round(n_layers / 2)
    with safe_open(VEC_ROOT / VEC_TAG[MODEL] / "diff.safetensors", framework="pt") as handle:
        diff = handle.get_tensor("diff")
    directions = diff[layer, FEATURES].float().to(device)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    log.info("layer %d/%d, %d features on %d prompts", layer, n_layers, len(FEATURES), len(prompts))

    manifest = []
    for idx, entry in enumerate(prompts, 1):
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
        hidden = out.hidden_states[layer][0].float()
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
        mu = cos[content].mean(0)
        sigma = cos[content].std(0).clamp_min(1e-6)
        z = (cos - mu) / sigma
        display = [cast("str", tok.decode([i])) for i in ids]

        folder = args.out_root / f"{idx:02d}"
        for col, fpid in enumerate(FEATURES):
            fc, fa = names[fpid]
            title = f"Feature: {fc}  ‖  {fa}   (pair {fpid})"
            sublines = [
                f"color:  red = {fc}   |   blue = {fa}   (z-scored cosine, white ≈ 0)",
                f"prompt {idx:02d}/{len(prompts)}: {entry['name']}   —   assistant LIES (prefill)",
                f"model: {MODEL.split('/')[-1]}   ·   layer {layer}/{n_layers} (½ depth)   ·   z-scored cosine",
            ]
            render_svg(
                display,
                z[:, col].tolist(),
                r0 if r0 >= 0 else 0,
                title,
                sublines,
                folder / f"{slug(fc + '--' + fa)}.svg",
            )
        manifest.append(
            {"index": f"{idx:02d}", "name": entry["name"], "tokens": len(ids), "reply_span": [int(r0), int(r1)]}
        )
        log.info(
            "prompt %02d: %s (%d tokens, reply[%d:%d]) -> %d SVGs", idx, entry["name"], len(ids), r0, r1, len(FEATURES)
        )

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    log.info("done: %d prompts x %d features -> %s", len(prompts), len(FEATURES), args.out_root)


if __name__ == "__main__":
    main()
