from __future__ import annotations

import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
from safetensors.torch import save_file

from exp_attn_interp.vectors import load_diff_vectors

log = logging.getLogger("orthovec")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--siblings", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("work/vectors_ortho.safetensors"))
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(4)

    from exp_attn_interp.config import vector_tag_for

    V = load_diff_vectors(device="cpu", tag=vector_tag_for(args.model))
    layers, pairs, hidden = V.shape
    ortho = torch.empty_like(V)
    kept = torch.empty(layers, pairs)
    for ell in range(layers):
        Vl = V[ell]
        cos = Vl @ Vl.T
        cos.fill_diagonal_(-2.0)
        sib = cos.topk(args.siblings, dim=1).indices
        out = torch.empty_like(Vl)
        kept_l = torch.empty(Vl.shape[0])
        for k in range(Vl.shape[0]):
            S = Vl[sib[k]]
            Q, _ = torch.linalg.qr(S.T)
            v = Vl[k]
            r = v - Q @ (Q.T @ v)
            n = r.norm()
            kept_l[k] = n
            out[k] = r / n.clamp_min(1e-8)
        ortho[ell], kept[ell] = out, kept_l
        if ell % 6 == 0 or ell == layers - 1:
            log.info(
                "layer %d: kept norm median %.3f (p10 %.3f, p90 %.3f)",
                ell,
                kept[ell].median(),
                kept[ell].quantile(0.1),
                kept[ell].quantile(0.9),
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"diff": ortho, "kept": kept},
        str(args.out),
        metadata={"siblings": str(args.siblings), "source": "diff/assistant, sibling-orthogonalized"},
    )
    log.info("wrote %s: %d siblings removed per concept", args.out, args.siblings)


if __name__ == "__main__":
    main()
