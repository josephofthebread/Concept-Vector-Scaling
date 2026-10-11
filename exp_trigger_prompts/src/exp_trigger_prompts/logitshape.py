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

log = logging.getLogger("logitshape")

MODEL = "Qwen/Qwen3-8B"
LAYER = 18
LOGIT_DESCS = ["entropy", "top1_minus_top2", "p2", "p3", "p4", "mass_234"]
FEATURE_METRICS = ["max_abs_z", "top1_minus_top2_z"]
SUBSETS = ["all", "content", "prompt", "response"]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation via argsort-of-argsort ranks; nan if fewer than 3 points or a flat series."""
    if a.size < 3:
        return float("nan")
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("data/behavioral_prompts.json"))
    parser.add_argument("--out", type=Path, default=Path("work/logit-corr/spearman.json"))
    args = parser.parse_args()

    prompts = json.loads(args.prompts.read_text())
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

    results = []
    for entry in prompts:
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

        probs = torch.softmax(out.logits[0].float(), dim=-1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
        topv = probs.topk(4, dim=-1).values
        logit = {
            "entropy": entropy,
            "top1_minus_top2": topv[:, 0] - topv[:, 1],
            "p2": topv[:, 1],
            "p3": topv[:, 2],
            "p4": topv[:, 3],
            "mass_234": topv[:, 1] + topv[:, 2] + topv[:, 3],
        }

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
        absz = z.abs()
        top2z = absz.topk(2, dim=-1).values
        feature = {"max_abs_z": top2z[:, 0], "top1_minus_top2_z": top2z[:, 0] - top2z[:, 1]}

        idx = {
            "all": np.arange(len(ids)),
            "content": np.where(content.numpy())[0],
            "prompt": np.arange(p0, p1) if p0 >= 0 else np.array([], dtype=int),
            "response": np.arange(r0, r1) if r0 >= 0 else np.array([], dtype=int),
        }
        feat_np = {k: v.cpu().numpy() for k, v in feature.items()}
        logit_np = {k: v.cpu().numpy() for k, v in logit.items()}

        rho: dict[str, dict[str, dict[str, float]]] = {}
        for fm in FEATURE_METRICS:
            rho[fm] = {}
            for sub in SUBSETS:
                sel = idx[sub]
                rho[fm][sub] = {ld: spearman(feat_np[fm][sel], logit_np[ld][sel]) for ld in LOGIT_DESCS}

        results.append(
            {
                "stem": entry["stem"],
                "name": entry["name"],
                "outcome": entry["outcome"],
                "n": {s: int(idx[s].size) for s in SUBSETS},
                "rho": rho,
            }
        )
        log.info(
            "%s: n_all=%d n_content=%d n_prompt=%d n_resp=%d",
            entry["stem"],
            idx["all"].size,
            idx["content"].size,
            idx["prompt"].size,
            idx["response"].size,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    log.info("wrote %d chats -> %s", len(results), args.out)


if __name__ == "__main__":
    main()
