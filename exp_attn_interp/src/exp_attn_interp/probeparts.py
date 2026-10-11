from __future__ import annotations

import logging
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from exp_attn_interp.common import load_model, story_span
from exp_attn_interp.config import GENDATA_ROOT, INSTRUCT_MODEL, vector_tag_for
from exp_attn_interp.decomp import POOL_MIN, POOL_SKIP
from exp_attn_interp.vectors import load_diff_vectors
from exp_attn_interp.write import story_ids

log = logging.getLogger("probeparts")
ROWS = (12, 22, 35)
PARTS = ("whole", "stance", "residual")


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--every", type=int, default=16)
    parser.add_argument("--per-pole", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--out", type=Path, default=Path("work/probeparts.npz"))
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    V = load_diff_vectors(device=str(model.device), tag=vector_tag_for(args.model))
    dev = model.device
    dirs = {}
    for L in ROWS:
        X = V[L]
        m = X.mean(0)
        mh = m / m.norm()
        Xc = X - m
        _, _, Wt = torch.linalg.svd(Xc, full_matrices=False)
        Q, _ = torch.linalg.qr(torch.cat([mh[None], Wt[: args.rank]], 0).T)
        st = (X @ Q) @ Q.T
        dirs[L] = {
            "whole": X,
            "stance": st / st.norm(dim=-1, keepdim=True).clamp_min(1e-8),
            "residual": (X - st) / (X - st).norm(dim=-1, keepdim=True).clamp_min(1e-8),
        }
    tbl = pq.read_table(
        GENDATA_ROOT / "data" / "semantic.parquet",
        columns=["pair_id", "concept_text", "antagonist_text"],
        filters=[("language_id", "==", 0)],
    )
    by_pair: dict[int, list[tuple[str, str]]] = {}
    for p, c, a in zip(tbl["pair_id"].to_pylist(), tbl["concept_text"].to_pylist(), tbl["antagonist_text"].to_pylist()):
        by_pair.setdefault(int(p), []).append((c, a))
    pairs = sorted(by_pair)[:: args.every]
    S = args.per_pole
    mean = np.full((len(pairs), 2, S, len(ROWS), len(PARTS)), np.nan, dtype=np.float32)
    std = np.full_like(mean, np.nan)
    for n, k in enumerate(pairs):
        for j, (c, a) in enumerate(by_pair[k][:S]):
            for side, text in ((0, c), (1, a)):
                ids = story_ids(tokenizer, text)
                s0, s1 = story_span(tokenizer, ids, text)
                if (s1 - s0 - POOL_SKIP) >= POOL_MIN:
                    s0 += POOL_SKIP
                out = model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True, use_cache=False)
                for r, L in enumerate(ROWS):
                    h = out.hidden_states[L][0, s0:s1].to(torch.float32)
                    for q, part in enumerate(PARTS):
                        pr = h @ dirs[L][part][k]
                        mean[n, side, j, r, q] = float(pr.mean())
                        std[n, side, j, r, q] = float(pr.std())
        if n % 16 == 0:
            log.info("%d/%d", n, len(pairs))
    np.savez(str(args.out), mean=mean, std=std, pairs=np.array(pairs), rows=np.array(ROWS), parts=np.array(PARTS))
    gap = mean[:, 0].mean(1) - mean[:, 1].mean(1)
    spread = np.sqrt(0.5 * (mean[:, 0].std(1) ** 2 + mean[:, 1].std(1) ** 2))
    noise = np.sqrt((std**2).mean((1, 2)))
    for r, L in enumerate(ROWS):
        for q, part in enumerate(PARTS):
            log.info(
                "row %d %-8s: pole gap %.2f, story SNR %.2f, token SNR %.2f",
                L,
                part,
                np.median(gap[:, r, q]),
                np.median(gap[:, r, q] / spread[:, r, q]),
                np.median(gap[:, r, q] / noise[:, r, q]),
            )


if __name__ == "__main__":
    main()
