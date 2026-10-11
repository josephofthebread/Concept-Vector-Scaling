from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from exp_attn_interp.config import NUM_BLOCKS, PAIRS

log = logging.getLogger("writean")


def specificity(mat: np.ndarray, covered: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Per covered concept: best head's value, and that head's median value on other concepts."""
    best, off, head_of = [], [], {}
    for k in covered:
        col = mat[:, k]
        h = int(np.nanargmax(col))
        head_of[int(k)] = h
        best.append(float(col[h]))
        row = mat[h, covered]
        row = row[np.isfinite(row) & (covered != k)]
        off.append(float(np.nanmedian(row)) if row.size else np.nan)
    return np.array(best), np.array(off), head_of


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--write-dir", type=Path, default=Path("work/write"))
    parser.add_argument("--ontology", type=Path, default=Path("../exp_gendata/data/semantic.ontology.json"))
    parser.add_argument("--out", type=Path, default=Path("work/writean.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    pairs = sorted(json.loads(args.ontology.read_text())["pairs"], key=lambda p: int(p["pair_id"]))
    files = sorted(args.write_dir.glob("item-*.safetensors"))
    if not files:
        raise SystemExit(f"no write outputs in {args.write_dir}")
    with safe_open(str(files[0]), framework="np") as h:
        heads = int(h.get_tensor("write_cos").shape[1])
    rows = NUM_BLOCKS * heads
    w_sum = np.zeros((rows, PAIRS))
    norm_sum = np.zeros(rows)
    mw_sum = np.zeros((rows, PAIRS))
    mw_n = np.zeros((rows, PAIRS))
    for f in files:
        with safe_open(str(f), framework="np") as h:
            w_sum += h.get_tensor("write_cos").astype(np.float32).reshape(rows, PAIRS)
            norm_sum += h.get_tensor("write_norm").astype(np.float32).reshape(rows)
            mw = h.get_tensor("marker_write").astype(np.float32).reshape(rows, PAIRS)
        v = ~np.isnan(mw)
        mw_sum += np.where(v, mw, 0.0)
        mw_n += v
    n = len(files)
    W = w_sum / n
    with np.errstate(invalid="ignore"):
        MW = np.where(mw_n > 0, mw_sum / np.clip(mw_n, 1, None), np.nan)
    activity = norm_sum / n

    Wc = W - W.mean(axis=1, keepdims=True)
    ev = np.clip(np.linalg.svd(Wc, compute_uv=False) ** 2, 0, None)
    dim = float(ev.sum() ** 2 / np.square(ev).sum()) if ev.sum() > 0 else 0.0
    s = np.linalg.svd(W - W.mean(axis=1, keepdims=True), compute_uv=False)
    energy = (s**2 / (s**2).sum())[:5]

    covered_w = np.arange(PAIRS)
    best_w, off_w, head_w = specificity(W, covered_w)
    gap_w = best_w - off_w

    covered_m = np.array([k for k in range(PAIRS) if np.isfinite(MW[:, k]).any()])
    best_m, off_m, head_m = specificity(MW, covered_m) if covered_m.size else (np.array([np.nan]),) * 2 + ({},)
    gap_m = best_m - off_m

    layer_profile = [
        round(float(np.abs(W[b * heads : (b + 1) * heads]).max(axis=1).mean()), 4) for b in range(NUM_BLOCKS)
    ]

    exemplars = []
    for j in np.argsort(-gap_w)[:12]:
        k = int(covered_w[j])
        h = head_w[k]
        b, hd = divmod(h, heads)
        mk = float(MW[h, k]) if np.isfinite(MW[h, k]) else None
        exemplars.append(
            {
                "pair": f"{pairs[k]['concept']} || {pairs[k]['antagonist']}"[:70],
                "block": b,
                "head": hd,
                "write": round(float(best_w[j]), 4),
                "offtarget": round(float(off_w[j]), 4),
                "marker_write_at_head": mk,
            }
        )

    summary: dict[str, Any] = {
        "items": n,
        "write_map_eff_dim": round(dim, 2),
        "write_map_energy_top5": [round(float(e), 3) for e in energy],
        "write_specificity": {
            "best_write_median": round(float(np.nanmedian(best_w)), 4),
            "offtarget_median": round(float(np.nanmedian(off_w)), 4),
            "gap_median": round(float(np.nanmedian(gap_w)), 4),
            "gap_p90": round(float(np.nanpercentile(gap_w, 90)), 4),
        },
        "marker_test": {
            "concepts_covered": int(covered_m.size),
            "best_marker_write_median": round(float(np.nanmedian(best_m)), 4),
            "offtarget_median": round(float(np.nanmedian(off_m)), 4),
            "gap_median": round(float(np.nanmedian(gap_m)), 4),
            "gap_p90": round(float(np.nanpercentile(gap_m, 90)), 4) if covered_m.size else None,
        },
        "layer_profile_max_write": layer_profile,
        "head_activity_median": round(float(np.median(activity)), 4),
        "exemplars": exemplars,
    }
    exemplars_marker = []
    if covered_m.size:
        for j in np.argsort(-gap_m)[:12]:
            k = int(covered_m[j])
            h = head_m[k]
            b, hd = divmod(h, heads)
            exemplars_marker.append(
                {
                    "pair": f"{pairs[k]['concept']} || {pairs[k]['antagonist']}"[:70],
                    "block": b,
                    "head": hd,
                    "marker_gap": round(float(gap_m[j]), 4),
                    "marker_best": round(float(best_m[j]), 4),
                    "write_at_head": round(float(W[h, k]), 4),
                    "n_items": int(mw_n[h, k]),
                }
            )
    summary["exemplars_marker"] = exemplars_marker

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    (args.out.parent / "per_concept_write.json").write_text(
        json.dumps(
            [
                {
                    "pair_id": int(k),
                    "best_write": round(float(best_w[i]), 4),
                    "offtarget": round(float(off_w[i]), 4),
                    "gap": round(float(gap_w[i]), 4),
                    "head": head_w[int(k)],
                }
                for i, k in enumerate(covered_w)
            ]
        )
    )
    log.info("write map eff dim=%.2f energy=%s", dim, summary["write_map_energy_top5"])
    ws = summary["write_specificity"]
    log.info(
        "write specificity: best=%.4f off=%.4f gap=%.4f (p90 %.4f)",
        ws["best_write_median"],
        ws["offtarget_median"],
        ws["gap_median"],
        ws["gap_p90"],
    )
    mt = summary["marker_test"]
    log.info(
        "marker test (%d concepts): best=%.4f off=%.4f gap=%.4f",
        mt["concepts_covered"],
        mt["best_marker_write_median"],
        mt["offtarget_median"],
        mt["gap_median"],
    )
    log.info("layer profile (max concept write per block): %s", layer_profile)
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
