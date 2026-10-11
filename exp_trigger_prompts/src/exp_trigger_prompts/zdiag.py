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

log = logging.getLogger("zdiag")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(VEC_TAG))
    parser.add_argument("--prefills", type=Path, default=Path("work/heldout_prefills_qwen.json"))
    parser.add_argument("--out-root", type=Path, default=Path("work/zdiag"))
    args = parser.parse_args()

    entries = json.loads(args.prefills.read_text())
    thinking: dict[str, Any] = {"enable_thinking": False} if "Qwen3" in args.model else {}
    attn = "eager" if "gemma" in args.model.lower() else "sdpa"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=attn, device_map="auto"
    )
    model.eval()
    device = model.device
    layer = round(model.config.num_hidden_layers / 2)
    with safe_open(VEC_ROOT / VEC_TAG[args.model] / "diff.safetensors", framework="pt") as handle:
        diff = handle.get_tensor("diff")
    directions = diff[layer].float().to(device)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    rows = []
    for entry in entries:
        messages = entry["conversation"]
        enc = cast(
            "Any",
            tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, return_tensors="pt", return_dict=True, **thinking
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
        if p0 < 0 or r0 < 0:
            log.warning("skip pair %s %s: span recovery failed", entry["pair_id"], entry["pole"])
            continue
        content = list(range(p0, p1)) + list(range(r0, r1))
        block = cos[content]
        z = (block - block.mean(0)) / block.std(0).clamp_min(1e-6)
        zc = z.cpu().numpy()

        n_p, n_r = p1 - p0, r1 - r0
        ranks = zc.argsort(0).argsort(0).astype(np.float64) + 1.0
        r_sum_prompt = ranks[:n_p].sum(0)
        auroc = (r_sum_prompt - n_p * (n_p + 1) / 2.0) / (n_p * n_r)

        rows.append(
            {
                "pair_id": entry["pair_id"],
                "pole": entry["pole"],
                "n_prompt": int(n_p),
                "n_reply": int(n_r),
                "mean_auroc": float(auroc.mean()),
                "median_auroc": float(np.median(auroc)),
                "frac_features_prompt_higher": float((auroc > 0.5).mean()),
                "mean_z_prompt": float(zc[:n_p].mean()),
                "mean_z_reply": float(zc[n_p:].mean()),
            }
        )
        log.info(
            "pair %s [%s] nP=%d nR=%d mean_auroc=%.3f frac>0.5=%.2f  zP=%+.2f zR=%+.2f",
            entry["pair_id"],
            entry["pole"],
            n_p,
            n_r,
            auroc.mean(),
            (auroc > 0.5).mean(),
            zc[:n_p].mean(),
            zc[n_p:].mean(),
        )

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / f"{args.model.split('/')[-1]}.json").write_text(json.dumps(rows, indent=2))
    aurocs = [r["mean_auroc"] for r in rows]
    log.info(
        "OVERALL %s: %d convs, mean feature-AUROC(prompt>reply) = %.3f (min %.3f, max %.3f)",
        args.model,
        len(rows),
        float(np.mean(aurocs)),
        min(aurocs),
        max(aurocs),
    )


if __name__ == "__main__":
    main()
