from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

log = logging.getLogger("decompan")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--decomp-dir", type=Path, default=Path("work/decomp"))
    parser.add_argument("--ontology", type=Path, default=Path("../exp_gendata/data/semantic.ontology.json"))
    parser.add_argument("--edgemap", type=Path, default=Path("work/edgemap.json"))
    parser.add_argument("--band-start", type=int, default=19, help="first block of the steering-optimum band")
    parser.add_argument("--out", type=Path, default=Path("work/decompan.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    pairs = sorted(json.loads(args.ontology.read_text())["pairs"], key=lambda p: int(p["pair_id"]))
    files = sorted(args.decomp_dir.glob("item-*.safetensors"))
    if not files:
        raise SystemExit(f"no decomp outputs in {args.decomp_dir}")
    with safe_open(str(files[0]), framework="np") as h:
        read_layers = [int(x) for x in h.metadata()["read_layers"].split(",")]
        shape = h.get_tensor("c_attn").shape
    n_blocks, heads = int(shape[1]), int(shape[2])
    nL = len(read_layers)
    iL = nL - 1
    pre = args.band_start - 1

    acc: dict[tuple[int, int], dict[str, Any]] = {}
    for f in files:
        with safe_open(str(f), framework="np") as h:
            k = int(h.get_tensor("pair_id"))
            side = int(h.get_tensor("side"))
            ca = h.get_tensor("c_attn").astype(np.float64)
            cm = h.get_tensor("c_mlp").astype(np.float64)
            ce = h.get_tensor("c_embed").astype(np.float64)
            tot = h.get_tensor("total").astype(np.float64)
        a = acc.setdefault((k, side), {"ca": 0.0, "cm": 0.0, "ce": 0.0, "tot": 0.0, "n": 0})
        a["ca"] = a["ca"] + ca
        a["cm"] = a["cm"] + cm
        a["ce"] = a["ce"] + ce
        a["tot"] = a["tot"] + tot
        a["n"] += 1
    concepts = sorted({k for k, _ in acc} & {k for k, s in acc if s == 1})
    log.info(
        "%d stories, %d concepts with both poles, %d blocks x %d heads", len(files), len(concepts), n_blocks, heads
    )

    exact, shares, f50, f90, fpre, sparsity = [], [], [], [], [], []
    f50_rel: dict[int, list[float]] = {}
    per_block_attn = np.zeros((len(concepts), n_blocks))
    per_block_mlp = np.zeros((len(concepts), n_blocks))
    top3: Counter = Counter()
    per_concept: list[dict[str, Any]] = []
    D_attn_all = np.zeros((len(concepts), n_blocks, heads))
    for ci, k in enumerate(concepts):
        c, a = acc[(k, 0)], acc[(k, 1)]
        Da = c["ca"] / c["n"] - a["ca"] / a["n"]
        Dm = c["cm"] / c["n"] - a["cm"] / a["n"]
        De = c["ce"] / c["n"] - a["ce"] / a["n"]
        Dt = c["tot"] / c["n"] - a["tot"] / a["n"]
        recon = De + Da.sum((1, 2)) + Dm.sum(1)
        exact.append(float(np.abs(recon - Dt).max() / max(np.abs(Dt).max(), 1e-9)))
        den = np.where(np.abs(Dt) > 1e-9, Dt, np.nan)
        shares.append(np.stack([Da.sum((1, 2)) / den, Dm.sum(1) / den, De / den]))
        for i, L in enumerate(read_layers):
            dti = Dt[i] if abs(Dt[i]) > 1e-9 else np.nan
            cumi = (De[i] + np.cumsum(Da[i].sum(1) + Dm[i])) / dti
            half = int(np.argmax(cumi[:L] >= 0.5)) if (cumi[:L] >= 0.5).any() else L
            f50_rel.setdefault(L, []).append(half / L)
        step = Da[iL].sum(1) + Dm[iL]
        dt = Dt[iL] if abs(Dt[iL]) > 1e-9 else np.nan
        cum = (De[iL] + np.cumsum(step)) / dt
        f50.append(int(np.argmax(cum >= 0.5)) if (cum >= 0.5).any() else n_blocks)
        f90.append(int(np.argmax(cum >= 0.9)) if (cum >= 0.9).any() else n_blocks)
        fpre.append(float(cum[pre]))
        per_block_attn[ci] = Da[iL].sum(1) / dt
        per_block_mlp[ci] = Dm[iL] / dt
        H = Da[iL]
        D_attn_all[ci] = H
        mags = np.abs(H).ravel()
        total = mags.sum()
        sparsity.append(float(total * total / np.square(mags).sum()) if total > 0 else 0.0)
        for idx in np.argsort(-H.ravel())[:3]:
            top3[int(idx)] += 1
        b0, h0 = divmod(int(np.argmax(H)), heads)
        top_heads = [
            [int(i // heads), int(i % heads), round(float(H.ravel()[i] / dt), 4) if np.isfinite(dt) else None]
            for i in np.argsort(-H.ravel())[:10]
        ]
        per_concept.append(
            {
                "pair_id": k,
                "top_head": [b0, h0],
                "top_heads": top_heads,
                "top_head_share": round(float(H.max() / dt), 4) if np.isfinite(dt) else None,
                "attn_share": round(float(Da[iL].sum() / dt), 4) if np.isfinite(dt) else None,
                "f50_block": f50[-1],
                "f90_block": f90[-1],
                "formed_before_band": round(fpre[-1], 4),
                "sparsity_pr": round(sparsity[-1], 1),
            }
        )

    shares_a = np.array(shares)
    med = lambda x: float(np.nanmedian(x))  # noqa: E731
    global_builders = [{"block": i // heads, "head": i % heads, "top3_count": c} for i, c in top3.most_common(15)]
    tot_attn = D_attn_all.sum((1, 2))[:, None, None]
    mean_share = np.nanmean(D_attn_all / np.where(np.abs(tot_attn) > 1e-9, tot_attn, np.nan), axis=0)

    detector_check: list[dict[str, Any]] = []
    if args.edgemap.exists():
        em = json.loads(args.edgemap.read_text())
        kidx = {k: i for i, k in enumerate(concepts)}
        name2k = {f"{pairs[k]['concept']} || {pairs[k]['antagonist']}"[:60]: k for k in concepts}
        for e in em.get("mid_stack_head_map", []):
            if e["n_concepts"] != 1:
                continue
            kk = name2k.get(e["concepts"][0]["pair"][:60])
            if kk is None or kk not in kidx:
                continue
            H = D_attn_all[kidx[kk]]
            rank = int((H.ravel() > H[e["block"], e["head"]]).sum()) + 1
            detector_check.append(
                {
                    "pair": f"{pairs[kk]['concept']} || {pairs[kk]['antagonist']}"[:60],
                    "block": e["block"],
                    "head": e["head"],
                    "builder_rank": rank,
                    "share": round(float(H[e["block"], e["head"]] / H.sum()), 4) if abs(H.sum()) > 1e-9 else None,
                }
            )

    cum_med = np.nanmedian(np.cumsum(per_block_attn + per_block_mlp, axis=1), axis=0) + med(shares_a[:, 2, iL])
    summary: dict[str, Any] = {
        "stories": len(files),
        "concepts": len(concepts),
        "read_layers": read_layers,
        "n_blocks": n_blocks,
        "band_start": args.band_start,
        "exactness_median_relerr": round(med(exact), 5),
        "exactness_p95_relerr": round(float(np.nanpercentile(exact, 95)), 5),
        "share_by_layer": {
            str(L): {
                "attention": round(med(shares_a[:, 0, i]), 3),
                "mlp": round(med(shares_a[:, 1, i]), 3),
                "embedding": round(med(shares_a[:, 2, i]), 3),
            }
            for i, L in enumerate(read_layers)
        },
        "formation": {
            "f50_block_median": int(np.median(f50)),
            "f90_block_median": int(np.median(f90)),
            "formed_before_band_median": round(med(fpre), 3),
            "formed_before_band_p10_p90": [
                round(float(np.nanpercentile(fpre, 10)), 3),
                round(float(np.nanpercentile(fpre, 90)), 3),
            ],
            "cum_curve_median": [round(float(x), 3) for x in cum_med],
        },
        "f50_relative_by_read_layer": {str(L): round(float(np.nanmedian(v)), 3) for L, v in f50_rel.items()},
        "per_block_attention_share_median": [round(float(x), 4) for x in np.nanmedian(per_block_attn, 0)],
        "per_block_mlp_share_median": [round(float(x), 4) for x in np.nanmedian(per_block_mlp, 0)],
        "builder_sparsity_pr_median": round(float(np.median(sparsity)), 1),
        "global_builder_heads_top3count": global_builders,
        "global_builder_heads_meanshare": [
            {"block": int(i // heads), "head": int(i % heads), "mean_share": round(float(v), 4)}
            for i, v in sorted(enumerate(mean_share.ravel()), key=lambda t: -t[1])[:12]
        ],
        "detector_heads_as_builders": detector_check,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    (args.out.parent / "per_concept_decomp.json").write_text(json.dumps(per_concept))

    print(f"\n=== EXACT DECOMPOSITION: {len(concepts)} concepts, {len(files)} stories, {n_blocks} blocks ===")
    print(
        f"exactness: median rel.err {summary['exactness_median_relerr']:.2e} (p95 {summary['exactness_p95_relerr']:.2e})"
    )
    print("share of the concept direction by read layer (attention / mlp / embedding):")
    for L, s in summary["share_by_layer"].items():
        print(f"  L{L:>2}: attn {s['attention']:+.3f}  mlp {s['mlp']:+.3f}  embed {s['embedding']:+.3f}")
    f = summary["formation"]
    print(
        f"formation (last layer): 50% by block {f['f50_block_median']}, 90% by block {f['f90_block_median']}; "
        f"assembled before steering band (block {args.band_start}): median {f['formed_before_band_median']:.3f} "
        f"(p10-p90 {f['formed_before_band_p10_p90']})"
    )
    print("cumulative formation (median):", f["cum_curve_median"])
    print(
        "half-formation point of each read layer's OWN vector, as a fraction of that layer's depth:",
        summary["f50_relative_by_read_layer"],
    )
    print(f"builder sparsity (PR over {n_blocks * heads} heads), median: {summary['builder_sparsity_pr_median']}")
    print(
        "global builder heads (top-3 count):", [(g["block"], g["head"], g["top3_count"]) for g in global_builders[:10]]
    )
    print(
        "global builder heads (mean share):",
        [(g["block"], g["head"], g["mean_share"]) for g in summary["global_builder_heads_meanshare"][:8]],
    )
    if detector_check:
        ranks = [d["builder_rank"] for d in detector_check]
        print(
            f"detector heads as builders of their own concept: {len(detector_check)} checked, median rank "
            f"{int(np.median(ranks))} of {n_blocks * heads}; in top-10: {sum(r <= 10 for r in ranks)}"
        )
        for d in detector_check[:8]:
            print(f"   L{d['block']:2d} H{d['head']:2d} rank={d['builder_rank']:4d} share={d['share']}  {d['pair']}")
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
