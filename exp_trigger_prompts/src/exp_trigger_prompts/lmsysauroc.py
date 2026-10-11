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

log = logging.getLogger("lmsysauroc")

MODEL = "Qwen/Qwen3-8B"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--sample", type=Path, default=Path("work/lmsys-auroc/sample.json"))
    parser.add_argument("--features", type=Path, default=Path("data/features.json"))
    parser.add_argument("--ontology", type=Path, default=Path("data/ontology.json"))
    parser.add_argument("--out", type=Path, default=Path("work/lmsys-auroc/scores.npz"))
    parser.add_argument("--max-tokens", type=int, default=320)
    args = parser.parse_args()

    sample = json.loads(args.sample.read_text())
    features = json.loads(args.features.read_text())["features"]
    onto = json.loads(args.ontology.read_text())["classes"]
    name2idx = {c["name"]: i for i, c in enumerate(onto)}
    concept_class = np.array([name2idx[f["class_name"]] for f in features], dtype=np.int16)

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

    scores = np.full((len(sample), len(layers), diff.shape[1]), np.nan, dtype=np.float32)
    class_id = np.array([it["class_id"] for it in sample], dtype=np.int16)
    kept = 0
    for i, it in enumerate(sample):
        messages = [{"role": "user", "content": it["user"]}]
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
        u0, u1 = locate_span(ids, it["user"], tok)
        if u0 < 0 or u1 - u0 < 2:
            continue
        sel = list(range(u0, u1))
        with torch.no_grad():
            out = model(ids_t.to(device), output_hidden_states=True)
        for li, layer in enumerate(layers):
            hidden = out.hidden_states[layer][0].float()
            h_unit = hidden / hidden.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            cos = h_unit[sel] @ unit[layer].T
            scores[i, li] = cos.mean(0).cpu().numpy().astype(np.float32)
        kept += 1
        if kept % 100 == 0:
            log.info("processed %d/%d convs", kept, len(sample))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        scores=scores,
        class_id=class_id,
        concept_class=concept_class,
        layers=np.array(layers, dtype=np.int16),
        kept=np.int32(kept),
    )
    log.info("wrote %s: %d/%d convs scored, scores %s", args.out, kept, len(sample), scores.shape)


if __name__ == "__main__":
    main()
