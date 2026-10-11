from __future__ import annotations

import logging
import time
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file

from exp_attn_interp.common import load_model, story_span
from exp_attn_interp.config import GENDATA_ROOT, INSTRUCT_MODEL
from exp_attn_interp.decomp import POOL_MIN, POOL_SKIP
from exp_attn_interp.vectors import load_diff_vectors
from exp_attn_interp.write import story_ids

log = logging.getLogger("storyproj")


@torch.inference_mode()
def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--per-pole", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path("work/storyproj"))
    parser.add_argument("--start", type=int, default=0, help="first pair index (in sorted order)")
    parser.add_argument("--end", type=int, default=0, help="end pair index (exclusive; 0 = all)")
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    dev = model.device
    G = load_diff_vectors(device=str(dev))
    tbl = pq.read_table(
        GENDATA_ROOT / "data" / "semantic.parquet",
        columns=["pair_id", "concept_text", "antagonist_text"],
        filters=[("language_id", "==", 0)],
    )
    by_pair: dict[int, list[tuple[str, str]]] = {}
    for p, c, a in zip(tbl["pair_id"].to_pylist(), tbl["concept_text"].to_pylist(), tbl["antagonist_text"].to_pylist()):
        by_pair.setdefault(int(p), []).append((c, a))
    started = time.monotonic()
    todo = sorted(by_pair)[args.start : args.end or None]
    for n, k in enumerate(todo):
        path = args.out / f"pair-{k:04d}.safetensors"
        if path.exists():
            continue
        v = G[:, k]
        N = min(args.per_pole, len(by_pair[k]))
        mean = torch.zeros(2, N, 37)
        std = torch.zeros(2, N, 37)
        cmean = torch.zeros(2, N, 37)
        cstd = torch.zeros(2, N, 37)
        pooled = torch.zeros(2, N, 37)
        for j, (c, a) in enumerate(by_pair[k][:N]):
            for side, text in ((0, c), (1, a)):
                ids = story_ids(tokenizer, text)
                s0, s1 = story_span(tokenizer, ids, text)
                span = list(range(s0 + POOL_SKIP, s1)) if (s1 - s0 - POOL_SKIP) >= POOL_MIN else list(range(s0, s1))
                out = model(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True, use_cache=False)
                H = torch.stack([h[0][span].to(torch.float32) for h in out.hidden_states])
                proj = (H * v[:, None, :]).sum(-1)
                cos = proj / H.norm(dim=-1).clamp_min(1e-6)
                mean[side, j] = proj.mean(1).cpu()
                std[side, j] = proj.std(1).cpu()
                cmean[side, j] = cos.mean(1).cpu()
                cstd[side, j] = cos.std(1).cpu()
                pooled[side, j] = (H.mean(1) * v).sum(-1).cpu()
        save_file(
            {"mean": mean, "std": std, "cmean": cmean, "cstd": cstd, "pooled": pooled},
            str(path),
            metadata={"pair": str(k)},
        )
        if n < 2 or n % 100 == 0:
            log.info(
                "%d/%d pair %d: %d stories/pole, %.1f min", n, len(by_pair), k, N, (time.monotonic() - started) / 60
            )
    log.info("done %.1f min", (time.monotonic() - started) / 60)


if __name__ == "__main__":
    main()
