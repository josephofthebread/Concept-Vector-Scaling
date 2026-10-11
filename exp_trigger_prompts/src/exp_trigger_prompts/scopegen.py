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

log = logging.getLogger("scopegen")

MODEL = "Qwen/Qwen3-8B"
METHOD = "cosine z-scored vs own tokens"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("data/behavioral_prompts.json"))
    parser.add_argument("--features", type=Path, default=Path("data/features.json"))
    parser.add_argument("--out-root", type=Path, default=Path("work/scope-data"))
    args = parser.parse_args()

    features = json.loads(args.features.read_text())["features"]
    assert [f["pair_id"] for f in features] == list(range(len(features))), "pair_id must be 0..N-1"
    n_pairs = len(features)
    prompts = json.loads(args.prompts.read_text())
    args.out_root.mkdir(parents=True, exist_ok=True)

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
    assert diff.shape[0] == n_layers + 1 and diff.shape[1] == n_pairs, diff.shape
    unit = diff / diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    mid = unit[n_layers // 2]
    sim = (mid @ mid.T).cpu().numpy()
    np.fill_diagonal(sim, -1.0)
    neighbors = [np.argsort(-row)[:8].tolist() for row in sim]
    concepts = {
        "pairs": [
            {"pair": f["pair_id"], "concept": f["concept"], "antagonist": f["antagonist"], "class": f["class_name"]}
            for f in features
        ],
        "neighbors": neighbors,
    }
    (args.out_root / "concepts.json").write_text(json.dumps(concepts, ensure_ascii=False))
    log.info("wrote concepts.json (%d pairs)", n_pairs)

    generations = []
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

        p0, p1 = locate_span(ids, messages[0]["content"], tok)
        r0, r1 = locate_span(ids, messages[1]["content"], tok, start=max(p1, 0))
        content = torch.zeros(len(ids), dtype=torch.bool)
        if p0 >= 0:
            content[p0:p1] = True
        if r0 >= 0:
            content[r0:r1] = True
        if content.sum() < 2:
            content[:] = True

        pieces = tok.convert_ids_to_tokens(ids)
        role = ["response" if (r0 <= i < r1) else "prompt" for i in range(len(ids))]
        norms: dict[str, list[float]] = {}
        stem = entry["stem"]
        scales = {}
        for layer in layers:
            hidden = out.hidden_states[layer][0].float()
            norms[f"L{layer}"] = [round(float(v), 2) for v in hidden.norm(dim=-1).tolist()]
            h_unit = hidden / hidden.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            cos = h_unit @ unit[layer].T
            block = cos[content]
            z = (cos - block.mean(0)) / block.std().clamp_min(1e-6)
            (args.out_root / f"{stem}.m0.L{layer}.bin").write_bytes(z.cpu().numpy().astype(np.float32).tobytes())
            scales[f"m0.L{layer}"] = 1.0

        (args.out_root / f"{stem}.tokens.json").write_text(
            json.dumps(
                {"pieces": pieces, "role": role, "logprob": [None] * len(ids), "norms": norms}, ensure_ascii=False
            )
        )
        generations.append(
            {
                "stem": stem,
                "behavior": entry["scenario"],
                "cell": entry["name"],
                "sample": str(idx),
                "steer": None,
                "topic": entry["name"],
                "tactic": "z vs own tokens",
                "half": "full",
                "tokens": len(ids),
                "width": max(p1, 0),
                "outcome": entry["outcome"],
                "ending": entry["outcome"],
                "variant": "prefill",
                "seed": 0,
                "scales": scales,
            }
        )
        log.info(
            "stem %s: %s (%d tokens, reply[%d:%d]) -> %d blobs", stem, entry["name"], len(ids), r0, r1, len(layers)
        )

    (args.out_root / "manifest.json").write_text(
        json.dumps(
            {
                "pairs": n_pairs,
                "methods": [METHOD],
                "layers": layers,
                "generations": generations,
            },
            ensure_ascii=False,
        )
    )
    log.info("wrote manifest.json (%d generations, %d layers) -> %s", len(generations), len(layers), args.out_root)


if __name__ == "__main__":
    main()
