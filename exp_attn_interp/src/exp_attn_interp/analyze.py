from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from exp_attn_interp.config import NUM_BLOCKS, PAIRS

log = logging.getLogger("analyze")


def participation_ratio(values: np.ndarray) -> float:
    values = np.clip(values, 0, None)
    total = values.sum()
    return float(total**2 / np.square(values).sum()) if total > 0 else 0.0


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--measure-dir", type=Path, default=Path("work/measure"))
    parser.add_argument("--ontology", type=Path, default=Path("../exp_gendata/data/semantic.ontology.json"))
    parser.add_argument("--out", type=Path, default=Path("work/analyze.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    pairs = sorted(json.loads(args.ontology.read_text())["pairs"], key=lambda p: int(p["pair_id"]))
    files = sorted(args.measure_dir.glob("item-*.safetensors"))
    if not files:
        raise SystemExit(f"no measure outputs in {args.measure_dir}")
    log.info("aggregating %d items", len(files))

    with safe_open(str(files[0]), framework="np") as h:
        heads = int(h.get_tensor("rho").shape[2])
    rows = NUM_BLOCKS * heads
    kind_sum: dict[str, np.ndarray] = {}
    kind_n: dict[str, int] = {}
    raw_footprint_sum = np.zeros((rows, PAIRS), np.float64)
    auc_sum = np.zeros((rows, PAIRS), np.float64)
    auc_n = np.zeros((rows, PAIRS), np.int32)
    n_footprint = 0

    for path in files:
        with safe_open(str(path), framework="np") as h:
            meta = json.loads(h.metadata()["positions"])
            rho = h.get_tensor("rho").astype(np.float32)
            rho_raw = h.get_tensor("rho_raw_assistant").astype(np.float32)
            marker_auc = h.get_tensor("marker_auc").astype(np.float32)
        kinds = [m["kind"] for m in meta]
        for idx, kind in enumerate(kinds):
            flat = rho[idx].reshape(rows, PAIRS)
            if kind not in kind_sum:
                kind_sum[kind] = np.zeros((rows, PAIRS), np.float64)
                kind_n[kind] = 0
            kind_sum[kind] += flat
            kind_n[kind] += 1
        if "role_assistant" in kinds:
            raw_footprint_sum += rho_raw.reshape(rows, PAIRS)
            n_footprint += 1
        present = marker_auc.reshape(rows, PAIRS)
        valid = ~np.isnan(present)
        auc_sum += np.where(valid, present, 0.0)
        auc_n += valid.astype(np.int32)

    def footprint_eff_dim(mat: np.ndarray) -> float:
        c = mat - mat.mean(axis=1, keepdims=True)
        return participation_ratio(np.linalg.svd(c, compute_uv=False) ** 2)

    eff_by_kind = {kind: round(footprint_eff_dim(kind_sum[kind] / kind_n[kind]), 2) for kind in sorted(kind_sum)}
    footprint = kind_sum.get("role_assistant", next(iter(kind_sum.values()))) / kind_n.get(
        "role_assistant", next(iter(kind_n.values()))
    )
    raw_footprint = raw_footprint_sum / max(n_footprint, 1)
    raw_centered = raw_footprint - raw_footprint.mean(axis=1, keepdims=True)
    raw_eff_dim = participation_ratio(np.linalg.svd(raw_centered, compute_uv=False) ** 2)
    centered = footprint - footprint.mean(axis=1, keepdims=True)

    cols = centered - centered.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(cols, axis=0, keepdims=True)
    unit = cols / np.clip(norm, 1e-12, None)
    sample = np.random.default_rng(0).choice(PAIRS, size=min(PAIRS, 400), replace=False)
    cosmat = np.abs(unit[:, sample].T @ unit[:, sample])
    triu = cosmat[np.triu_indices_from(cosmat, k=1)]
    hist, edges = np.histogram(triu, bins=20, range=(0, 1))

    svals = np.linalg.svd(centered, compute_uv=False)
    energy = svals**2 / (svals**2).sum()
    eff_dim = participation_ratio(svals**2)

    absf = np.abs(centered)
    concentration = np.array([participation_ratio(absf[:, k]) for k in range(PAIRS)])
    mean_profile = centered.mean(axis=1)
    mp_unit = mean_profile / max(np.linalg.norm(mean_profile), 1e-12)
    specificity = 1 - np.abs(unit.T @ mp_unit)

    with np.errstate(invalid="ignore"):
        auc_mean = np.where(auc_n > 0, auc_sum / np.clip(auc_n, 1, None), np.nan)
    covered = np.array([k for k in range(PAIRS) if np.isfinite(auc_mean[:, k]).any()])
    best_head = {}
    best_auc = []
    pos_head = {}
    pos_auc = []
    pos_offtarget = []
    for k in covered:
        col = auc_mean[:, k]
        hbest = int(np.nanargmax(np.abs(col - 0.5)))
        best_head[k] = hbest
        best_auc.append(float(col[hbest]))
        hpos = int(np.nanargmax(col))
        pos_head[k] = hpos
        pos_auc.append(float(col[hpos]))
        row = auc_mean[hpos, covered]
        row = row[np.isfinite(row) & (covered != k)]
        pos_offtarget.append(float(np.nanmedian(row)) if row.size else 0.5)
    best_auc_arr = np.array(best_auc) if best_auc else np.array([np.nan])
    pos_auc_arr = np.array(pos_auc) if pos_auc else np.array([np.nan])
    pos_off_k = {int(covered[i]): pos_offtarget[i] for i in range(len(covered))}
    off = []
    for k, hbest in best_head.items():
        row = auc_mean[hbest, covered]
        row = row[np.isfinite(row) & (covered != k)]
        if row.size:
            off.append(float(np.nanmedian(np.abs(row - 0.5))))
    specific_gap = float(np.nanmedian(np.abs(best_auc_arr - 0.5))) - (float(np.median(off)) if off else np.nan)
    pos_gap = float(np.nanmedian(pos_auc_arr)) - (float(np.median(pos_offtarget)) if pos_offtarget else np.nan)

    summary: dict[str, Any] = {
        "items": len(files),
        "footprint_items": n_footprint,
        "footprint_eff_dim": round(eff_dim, 2),
        "footprint_eff_dim_raw": round(raw_eff_dim, 2),
        "footprint_eff_dim_by_kind": eff_by_kind,
        "footprint_energy_top5": [round(float(e), 3) for e in energy[:5]],
        "footprint_cos_hist": {"edges": [round(float(e), 2) for e in edges], "counts": [int(c) for c in hist]},
        "footprint_cos_median": round(float(np.median(triu)), 3),
        "concentration_median": round(float(np.median(concentration)), 1),
        "specificity_median": round(float(np.median(specificity)), 3),
        "marker_bridge": {
            "concepts_covered": int(covered.size),
            "pos_head_auc_median": round(float(np.nanmedian(pos_auc_arr)), 3),
            "pos_head_auc_p90": round(float(np.nanpercentile(pos_auc_arr, 90)), 3),
            "pos_head_offtarget_median": round(float(np.median(pos_offtarget)), 3) if pos_offtarget else None,
            "pos_head_specificity_gap": round(pos_gap, 3),
            "best_head_auc_median": round(float(np.nanmedian(best_auc_arr)), 3),
            "best_head_auc_p90": round(float(np.nanpercentile(np.abs(best_auc_arr - 0.5) + 0.5, 90)), 3)
            if best_auc
            else None,
            "best_head_specificity_gap": round(specific_gap, 3),
        },
    }
    exemplars = []
    if pos_auc:
        score = pos_auc_arr - np.array([pos_off_k[int(covered[i])] for i in range(len(covered))])
        for j in np.argsort(-score)[:12]:
            k = int(covered[j])
            hpos = pos_head[k]
            b, hd = divmod(hpos, heads)
            exemplars.append(
                {
                    "pair": f"{pairs[k]['concept']} || {pairs[k]['antagonist']}"[:80],
                    "block": b,
                    "head": hd,
                    "auc": round(pos_auc[j], 3),
                    "offtarget": round(float(pos_off_k.get(k, 0.5)), 3),
                    "n_items": int(auc_n[hpos, k]),
                }
            )
    summary["marker_bridge"]["exemplars"] = exemplars

    per_concept = []
    for i, k in enumerate(covered):
        k = int(k)
        hpos = pos_head[k]
        per_concept.append(
            {
                "pair_id": k,
                "pos_auc": round(pos_auc[i], 4),
                "offtarget": round(pos_off_k.get(k, 0.5), 4) if pos_auc else None,
                "gap": round(pos_auc[i] - pos_off_k.get(k, 0.5), 4) if pos_auc else None,
                "best_head": hpos,
                "n_items": int(auc_n[hpos, k]),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    (args.out.parent / "per_concept.json").write_text(json.dumps(per_concept))
    log.info("footprint effective dim: assistant fixed=%.2f raw=%.2f (dead exp ~3)", eff_dim, raw_eff_dim)
    log.info("footprint eff dim by position kind: %s", eff_by_kind)
    log.info("footprint |cos| median=%.3f", summary["footprint_cos_median"])
    mb = summary["marker_bridge"]
    log.info(
        "marker bridge (%d concepts): best-ATTEND head AUC median=%.3f (p90 %.3f), "
        "off-target=%.3f, specificity gap=%.3f",
        covered.size,
        mb["pos_head_auc_median"],
        mb["pos_head_auc_p90"],
        mb["pos_head_offtarget_median"],
        mb["pos_head_specificity_gap"],
    )
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
