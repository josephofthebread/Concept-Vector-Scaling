from __future__ import annotations

import json
import logging
import unicodedata
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import torch

from exp_attn_interp.common import load_model
from exp_attn_interp.config import MARKER_TOPN, MARKERS_LOGITLENS, READ_LAYER
from exp_attn_interp.vectors import load_diff_vectors

log = logging.getLogger("markers")


def latin_topn(scores: torch.Tensor, latin_mask: torch.Tensor, topn: int) -> list[list[int]]:
    """Top-N token ids per row, restricted to Latin-script tokens."""
    masked = scores.masked_fill(~latin_mask, float("-inf"))
    return masked.topk(topn, dim=-1).indices.cpu().tolist()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--layer", type=int, default=READ_LAYER)
    parser.add_argument("--topn", type=int, default=MARKER_TOPN)
    parser.add_argument("--out", type=Path, default=MARKERS_LOGITLENS)
    parser.add_argument("--latin-only", action=BooleanOptionalAction, default=True)
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model, tokenizer = load_model(offline=args.offline)
    device = model.device
    vectors = load_diff_vectors(device=str(device))[args.layer]
    gamma = model.model.norm.weight.to(torch.float32).to(device)
    unembed = model.lm_head.weight.to(torch.float32)

    vocab = unembed.shape[0]
    if args.latin_only:
        is_latin = []
        for i in range(vocab):
            letters = [c for c in tokenizer.decode([i]) if c.isalpha()]
            is_latin.append(
                any(c.isascii() for c in letters) and all("LATIN" in unicodedata.name(c, "") for c in letters)
            )
        latin = torch.tensor(is_latin, device=device)
        log.info("Latin-script tokens: %d / %d", int(latin.sum()), vocab)
    else:
        latin = torch.ones(vocab, dtype=torch.bool, device=device)

    with torch.inference_mode():
        logits = (vectors * gamma) @ unembed.T
        top = latin_topn(logits, latin, args.topn)
        bottom = latin_topn(-logits, latin, args.topn)

    payload = {
        "source": "logit_lens_latin" if args.latin_only else "logit_lens",
        "model": model.config._name_or_path,
        "layer": args.layer,
        "topn": args.topn,
        "concept": {str(k): ids for k, ids in enumerate(top)},
        "antagonist": {str(k): ids for k, ids in enumerate(bottom)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload))
    log.info(
        "wrote %s: %d concepts x top-%d Latin marker tokens at layer %d", args.out, len(top), args.topn, args.layer
    )


if __name__ == "__main__":
    main()
