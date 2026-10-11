from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from exp_attn_interp.config import vector_tag_for
from exp_attn_interp.vectors import load_diff_vectors

log = logging.getLogger("residrel")


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--a", type=Path, default=Path("work/vdecomp"))
    parser.add_argument("--b", type=Path, default=Path("work/vdecomp2"))
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("work/residrel.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dev = args.device
    V = load_diff_vectors(device=dev, tag=vector_tag_for("Qwen/Qwen3-8B"))
    rows = (12, 22, 35)
    proj = {}
    for L in rows:
        X = V[L]
        m = X.mean(0)
        mh = m / m.norm()
        _, _, Wt = torch.linalg.svd(X - m, full_matrices=False)
        P = Wt[: args.rank]
        proj[L] = (P, mh)
    files = sorted(f for f in args.a.glob("pair-*.safetensors") if (args.b / f.name).exists())[:: args.every]
    res: dict[int, dict[str, list[float]]] = {L: {"whole": [], "stance": [], "residual": [], "share": []} for L in rows}
    for i, f in enumerate(files):
        ta, tb = load_file(str(f)), load_file(str(args.b / f.name))
        vs = []
        for t in (ta, tb):
            A = (t["pole_head"][0] - t["pole_head"][1]).to(dev, torch.float32).sum(1)
            M = (t["pole_mlp"][0] - t["pole_mlp"][1]).to(dev, torch.float32)
            De = (t["pole_embed"][0] - t["pole_embed"][1]).to(dev, torch.float32)
            Vh = torch.cat([De[None], De[None] + torch.cumsum(A + M, 0)], 0)
            vs.append(Vh)
        for L in rows:
            P, mh = proj[L]
            a, b = vs[0][L], vs[1][L]
            sa = (a @ P.T) @ P + (a @ mh) * mh
            sb = (b @ P.T) @ P + (b @ mh) * mh
            ra, rb = a - sa, b - sb
            res[L]["whole"].append(float(unit(a) @ unit(b)))
            res[L]["stance"].append(float(unit(sa) @ unit(sb)))
            res[L]["residual"].append(float(unit(ra) @ unit(rb)))
            res[L]["share"].append(float(ra.norm() ** 2 / a.norm() ** 2))
        if i % 200 == 0:
            log.info("%d/%d", i, len(files))
    out = {}
    for L in rows:
        out[L] = {k: float(np.median(v)) for k, v in res[L].items()}
        log.info(
            "row %d: split-half cos whole %.3f | stance part %.3f | residual %.3f | residual energy share %.2f",
            L,
            out[L]["whole"],
            out[L]["stance"],
            out[L]["residual"],
            out[L]["share"],
        )
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
