from __future__ import annotations

import json
import logging
import random
import time
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from exp_attn_interp.common import load_model, story_span
from exp_attn_interp.config import GENDATA_ROOT, INSTRUCT_MODEL, vector_tag_for
from exp_attn_interp.decomp import POOL_MIN, POOL_SKIP, read_layers_for
from exp_attn_interp.vectors import load_diff_vectors
from exp_attn_interp.write import story_ids

log = logging.getLogger("ablate")


class HeadAblator:
    """Zero selected heads' contributions via pre-hooks on each block's o_proj."""

    def __init__(self, model: Any, head_dim: int) -> None:
        self.hd = head_dim
        self.active: dict[int, list[int]] = {}
        for b, block in enumerate(model.layers):
            block.self_attn.o_proj.register_forward_pre_hook(self._hook(b))

    def _hook(self, b: int) -> Any:
        def pre(_m: Any, inputs: tuple) -> tuple | None:
            heads = self.active.get(b)
            if not heads:
                return None
            x = inputs[0].clone()
            for h in heads:
                x[..., h * self.hd : (h + 1) * self.hd] = 0
            return (x,) + tuple(inputs[1:])

        return pre

    def set(self, heads: list[tuple[int, int]]) -> None:
        self.active = {}
        for b, h in heads:
            self.active.setdefault(b, []).append(h)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default=INSTRUCT_MODEL)
    parser.add_argument("--from-decomp", type=Path, default=Path("work/per_concept_decomp.json"))
    parser.add_argument(
        "--pairs", default="", help="comma list of pair ids; default: first --n-pairs of the decomp file"
    )
    parser.add_argument("--n-pairs", type=int, default=48)
    parser.add_argument("--top", type=int, default=5, help="ablate each concept's top-N builder heads")
    parser.add_argument("--heads", default="", help="explicit 'b:h,b:h' (single-concept use)")
    parser.add_argument("--per-pole", type=int, default=4)
    parser.add_argument("--read-frac", type=float, default=1.0)
    parser.add_argument("--controls", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("work/ablate.json"))
    parser.add_argument("--offline", action=BooleanOptionalAction, default=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model, tokenizer = load_model(offline=args.offline, model_name=args.model)
    cfg = model.config
    heads_n = int(cfg.num_attention_heads)
    hd = int(getattr(cfg, "head_dim", cfg.hidden_size // heads_n))
    n_blocks = len(model.model.layers)
    L = read_layers_for(n_blocks)[min(len(read_layers_for(n_blocks)) - 1, int(round(args.read_frac * 5)))]
    vectors = load_diff_vectors(device=str(model.device), tag=vector_tag_for(args.model))
    ablator = HeadAblator(model.model, hd)

    dec = json.loads(args.from_decomp.read_text()) if args.from_decomp.exists() else []
    top_of = {d["pair_id"]: [(b, h) for b, h, _ in d.get("top_heads", [])[: args.top]] for d in dec}
    if args.heads:
        explicit = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in args.heads.split(",")]
    pairs = [int(x) for x in args.pairs.split(",") if x] or sorted(top_of)[: args.n_pairs]

    tbl = pq.read_table(
        GENDATA_ROOT / "data" / "semantic.parquet",
        columns=["pair_id", "concept_text", "antagonist_text"],
        filters=[("language_id", "==", 0)],
    )
    by_pair: dict[int, list[tuple[str, str]]] = {}
    for p, c, a in zip(tbl["pair_id"].to_pylist(), tbl["concept_text"].to_pylist(), tbl["antagonist_text"].to_pylist()):
        by_pair.setdefault(int(p), []).append((c, a))

    def score_stories(k: int, stories: list[tuple[str, int]]) -> np.ndarray:
        v = vectors[L][k]
        out = []
        for story, _ in stories:
            ids = story_ids(tokenizer, story)
            s0, s1 = story_span(tokenizer, ids, story)
            span = list(range(s0 + POOL_SKIP, s1)) if (s1 - s0 - POOL_SKIP) >= POOL_MIN else list(range(s0, s1))
            with torch.inference_mode():
                hs = model(
                    input_ids=torch.tensor([ids], device=model.device), output_hidden_states=True, use_cache=False
                ).hidden_states[L][0]
            out.append(float(hs[span].to(torch.float32).mean(0) @ v))
        return np.array(out)

    rng = random.Random(0)
    results = []
    started = time.monotonic()
    for n, k in enumerate(pairs, start=1):
        rows = by_pair[k][: args.per_pole]
        stories = [(c, 1) for c, _ in rows] + [(a, 0) for _, a in rows]
        labels = np.array([lab for _, lab in stories])
        target = explicit if args.heads else top_of.get(k, [])
        if not target:
            continue
        ablator.set([])
        base = score_stories(k, stories)
        ablator.set(target)
        abl = score_stories(k, stories)
        D_base = base[labels == 1].mean() - base[labels == 0].mean()
        D_abl = abl[labels == 1].mean() - abl[labels == 0].mean()
        ctrl_drops: list[float] = []
        per_block: dict[int, int] = {}
        for b, _ in target:
            per_block[b] = per_block.get(b, 0) + 1
        for _ in range(args.controls):
            ctrl = [
                (b, h)
                for b, cnt in per_block.items()
                for h in rng.sample([h for h in range(heads_n) if (b, h) not in target], cnt)
            ]
            ablator.set(ctrl)
            c = score_stories(k, stories)
            D_c = c[labels == 1].mean() - c[labels == 0].mean()
            ctrl_drops.append(1 - D_c / D_base if abs(D_base) > 1e-9 else float("nan"))
        ablator.set([])
        drop = 1 - D_abl / D_base if abs(D_base) > 1e-9 else float("nan")
        cm, cs = float(np.nanmean(ctrl_drops)), float(np.nanstd(ctrl_drops) + 1e-9)
        results.append(
            {
                "pair_id": k,
                "heads": target,
                "D_base": round(float(D_base), 4),
                "D_ablated": round(float(D_abl), 4),
                "drop": round(float(drop), 4),
                "control_drop_mean": round(cm, 4),
                "control_drop_sd": round(cs, 4),
                "z_vs_control": round((drop - cm) / cs, 2),
                "auroc_base": round(auroc(base, labels), 3),
                "auroc_ablated": round(auroc(abl, labels), 3),
            }
        )
        log.info(
            "%d/%d pair %d: drop=%.3f ctrl=%.3f±%.3f z=%.1f auroc %.2f->%.2f",
            n,
            len(pairs),
            k,
            drop,
            cm,
            cs,
            results[-1]["z_vs_control"],
            results[-1]["auroc_base"],
            results[-1]["auroc_ablated"],
        )
    drops = np.array([r["drop"] for r in results])
    zs = np.array([r["z_vs_control"] for r in results])
    summary = {
        "model": args.model,
        "read_layer": L,
        "top": args.top,
        "controls": args.controls,
        "n": len(results),
        "drop_median": round(float(np.nanmedian(drops)), 4),
        "control_drop_median": round(float(np.nanmedian([r["control_drop_mean"] for r in results])), 4),
        "z_median": round(float(np.nanmedian(zs)), 2),
        "frac_z_gt_2": round(float(np.mean(zs > 2)), 3),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    log.info(
        "ablation: n=%d, drop median %.3f vs control %.3f, z median %.1f, frac z>2 %.2f (%.1f min)",
        len(results),
        summary["drop_median"],
        summary["control_drop_median"],
        summary["z_median"],
        summary["frac_z_gt_2"],
        (time.monotonic() - started) / 60,
    )


if __name__ == "__main__":
    main()
