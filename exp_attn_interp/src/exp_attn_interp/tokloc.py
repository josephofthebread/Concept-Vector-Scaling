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

log = logging.getLogger("tokloc")
ROWS = (12, 22, 35)
PARTS = ("whole", "stance", "residual")


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--every", type=int, default=16)
    parser.add_argument("--per-pole", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--out", type=Path, default=Path("work/tokloc.npz"))
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
        _, _, Wt = torch.linalg.svd(X - m, full_matrices=False)
        Q, _ = torch.linalg.qr(torch.cat([(m / m.norm())[None], Wt[: args.rank]], 0).T)
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
    out = np.full((len(pairs), 2, S, len(ROWS), len(PARTS), 3), np.nan, dtype=np.float32)
    corr = np.full((len(pairs), 2, S, len(ROWS)), np.nan, dtype=np.float32)
    for n, k in enumerate(pairs):
        for j, (c, a) in enumerate(by_pair[k][:S]):
            for side, text in ((0, c), (1, a)):
                ids = story_ids(tokenizer, text)
                s0, s1 = story_span(tokenizer, ids, text)
                if (s1 - s0 - POOL_SKIP) >= POOL_MIN:
                    s0 += POOL_SKIP
                hs = model(
                    input_ids=torch.tensor([ids], device=dev), output_hidden_states=True, use_cache=False
                ).hidden_states
                for r, L in enumerate(ROWS):
                    h = hs[L][0, s0:s1].to(torch.float32)
                    prs = {part: h @ dirs[L][part][k] for part in PARTS}
                    for q, part in enumerate(PARTS):
                        pr = prs[part]
                        pc = pr - pr.mean()
                        ktop = int(max(1, round(0.1 * pc.numel())))
                        top = torch.topk(pc.abs(), ktop).values.sum() / pc.abs().sum().clamp_min(1e-8)
                        kurt = (pc**4).mean() / (pc**2).mean().clamp_min(1e-8) ** 2 - 3
                        ac = (pc[:-1] * pc[1:]).mean() / (pc**2).mean().clamp_min(1e-8)
                        out[n, side, j, r, q] = float(top), float(kurt), float(ac)
                    a_, b_ = prs["stance"] - prs["stance"].mean(), prs["residual"] - prs["residual"].mean()
                    corr[n, side, j, r] = float((a_ * b_).mean() / (a_.std() * b_.std()).clamp_min(1e-8))
        if n % 16 == 0:
            log.info("%d/%d", n, len(pairs))
    np.savez(str(args.out), out=out, corr=corr, pairs=np.array(pairs), rows=np.array(ROWS), parts=np.array(PARTS))  # type: ignore[arg-type]
    for r, L in enumerate(ROWS):
        for q, part in enumerate(PARTS):
            v = np.nanmedian(out[:, :, :, r, q, :].reshape(-1, 3), 0)
            log.info("row %d %-8s: top-10%% share %.3f, excess kurtosis %.2f, lag-1 autocorr %.3f", L, part, *v)
        log.info("row %d per-token corr(stance, residual) median %.3f", L, float(np.nanmedian(corr[:, :, :, r])))


if __name__ == "__main__":
    main()
