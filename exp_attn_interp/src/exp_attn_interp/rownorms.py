from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp_attn_interp.common import story_span
from exp_attn_interp.config import GENVECTORS_ROOT
from exp_attn_interp.decomp import POOL_MIN, POOL_SKIP
from exp_attn_interp.xdecomp import render

log = logging.getLogger("rownorms")


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--presentation", default="assistant", choices=["assistant", "raw"])
    parser.add_argument("--every", type=int, default=16)
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    short = args.model.split("/")[-1]
    out = Path(f"work-x/{short}/rownorms.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.offline, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=args.offline,
        device_map={"": "cuda"},
    ).eval()
    final = {}
    model.model.norm.register_forward_pre_hook(lambda _m, inputs: final.__setitem__("x", inputs[0].detach()))
    B = len(model.model.layers)
    tbl = pq.read_table(
        GENVECTORS_ROOT / "data" / "semantic.parquet",
        columns=["pair_id", "concept_text", "antagonist_text"],
        filters=[("language_id", "==", 0)],
    )
    by_pair: dict[int, tuple[str, str]] = {}
    for p, c, a in zip(tbl["pair_id"].to_pylist(), tbl["concept_text"].to_pylist(), tbl["antagonist_text"].to_pylist()):
        by_pair.setdefault(int(p), (c, a))
    pairs = sorted(by_pair)[:: args.every]
    norms = [[] for _ in range(B + 1)]
    n_tok = 0
    for k in pairs:
        for text in by_pair[k]:
            ids = render(tokenizer, text, args.presentation)
            s0, s1 = story_span(tokenizer, ids, text)
            span = list(range(s0 + POOL_SKIP, s1)) if (s1 - s0 - POOL_SKIP) >= POOL_MIN else list(range(s0, s1))
            hs = model(
                input_ids=torch.tensor([ids], device=model.device), output_hidden_states=True, use_cache=False
            ).hidden_states
            for r in range(B):
                norms[r].append(hs[r][0, span].to(torch.float32).norm(dim=-1).cpu().numpy())
            norms[B].append(final["x"][0, span].to(torch.float32).norm(dim=-1).cpu().numpy())
            n_tok += len(span)
    med = [float(np.median(np.concatenate(x))) for x in norms]
    mean = [float(np.mean(np.concatenate(x))) for x in norms]
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "n_blocks": B,
                "n_stories": 2 * len(pairs),
                "n_tokens": n_tok,
                "median_token_norm": med,
                "mean_token_norm": mean,
            }
        )
    )
    log.info(
        "%s: %d stories, %d tokens; median token norm by row: %s",
        short,
        2 * len(pairs),
        n_tok,
        [round(m, 1) for m in med],
    )


if __name__ == "__main__":
    main()
