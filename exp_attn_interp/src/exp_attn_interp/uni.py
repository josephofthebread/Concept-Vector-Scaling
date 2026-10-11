from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from safetensors import safe_open

log = logging.getLogger("uni")
UNIVERSAL = [(30, 0), (27, 8), (34, 19), (34, 15), (30, 20), (26, 27)]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--decomp-dir", type=Path, default=Path("work/decomp"))
    parser.add_argument("--heads", default=",".join(f"{b}:{h}" for b, h in UNIVERSAL))
    parser.add_argument("--out", type=Path, default=Path("work/uni.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    heads = [tuple(int(x) for x in s.split(":")) for s in args.heads.split(",")]
    files = sorted(args.decomp_dir.glob("item-*.safetensors"))
    with safe_open(str(files[0]), framework="np") as h:
        read_layers = [int(x) for x in h.metadata()["read_layers"].split(",")]
    nL = len(read_layers)
    acc: dict[tuple[int, int], dict] = {}
    for f in files:
        with safe_open(str(f), framework="np") as h:
            k = int(h.get_tensor("pair_id"))
            side = int(h.get_tensor("side"))
            ca = h.get_tensor("c_attn").astype(np.float64)
            tot = h.get_tensor("total").astype(np.float64)
        a = acc.setdefault((k, side), {"ca": np.zeros_like(ca), "tot": np.zeros(nL), "n": 0})
        a["ca"] += ca
        a["tot"] += tot
        a["n"] += 1
    pairs = sorted({k for k, _ in acc} & {k for k, s in acc if s == 1})
    pairs = [k for k in pairs if (k, 0) in acc and (k, 1) in acc]
    out: dict = {"read_layers": read_layers, "n_concepts": len(pairs), "heads": {}}
    li_last, li_mid = nL - 1, nL // 2
    for b, hh in heads:
        rows = []
        for k in pairs:
            c, a = acc[(k, 0)], acc[(k, 1)]
            cc, ca_ = c["ca"] / c["n"], a["ca"] / a["n"]
            tc, ta = c["tot"] / c["n"], a["tot"] / a["n"]
            D = cc - ca_
            Dt = tc - ta
            rows.append(
                {
                    "pair": k,
                    "share_last": float(D[li_last, b, hh] / Dt[li_last]),
                    "share_mid": float(D[li_mid, b, hh] / Dt[li_mid]),
                    "pole_c": float(cc[li_last, b, hh]),
                    "pole_a": float(ca_[li_last, b, hh]),
                    "tot_c": float(tc[li_last]),
                    "tot_a": float(ta[li_last]),
                }
            )
        s = np.array([r["share_last"] for r in rows])
        sm = np.array([r["share_mid"] for r in rows])
        pc = np.array([r["pole_c"] for r in rows])
        pa = np.array([r["pole_a"] for r in rows])
        both = float(np.mean((pc > 0) & (pa < 0)))
        res = {
            "share_last_median": float(np.median(s)),
            "share_last_p10": float(np.percentile(s, 10)),
            "share_last_p90": float(np.percentile(s, 90)),
            "frac_positive_last": float(np.mean(s > 0)),
            "cv_last": float(np.std(s) / max(abs(np.mean(s)), 1e-9)),
            "share_mid_median": float(np.median(sm)),
            "frac_positive_mid": float(np.mean(sm > 0)),
            "frac_pushes_both_poles": both,
            "pole_c_median": float(np.median(pc)),
            "pole_a_median": float(np.median(pa)),
            "top5_concepts": sorted(rows, key=lambda r: -r["share_last"])[:5],
        }
        out["heads"][f"{b}:{hh}"] = res
        log.info(
            "L%d H%d: share@last median %.4f [p10 %.4f p90 %.4f] positive %.2f  cv %.2f | share@mid %.4f "
            "positive %.2f | pushes both poles %.2f (pole_c %.3f, pole_a %.3f)",
            b,
            hh,
            res["share_last_median"],
            res["share_last_p10"],
            res["share_last_p90"],
            res["frac_positive_last"],
            res["cv_last"],
            res["share_mid_median"],
            res["frac_positive_mid"],
            both,
            res["pole_c_median"],
            res["pole_a_median"],
        )
    rng = np.random.default_rng(0)
    B, H = acc[(pairs[0], 0)]["ca"].shape[1:]
    base: list[dict] = []
    for _ in range(50):
        b, hh = int(rng.integers(19, B)), int(rng.integers(0, H))
        ss: list[float] = []
        for k in pairs:
            c, a = acc[(k, 0)], acc[(k, 1)]
            D = c["ca"][li_last, b, hh] / c["n"] - a["ca"][li_last, b, hh] / a["n"]
            Dt = c["tot"][li_last] / c["n"] - a["tot"][li_last] / a["n"]
            ss.append(float(D / Dt))
        s_arr = np.array(ss)
        base.append({"head": [b, hh], "median": float(np.median(s_arr)), "frac_positive": float(np.mean(s_arr > 0))})
    out["random_late_heads"] = {
        "median_of_medians": float(np.median([x["median"] for x in base])),
        "median_frac_positive": float(np.median([x["frac_positive"] for x in base])),
    }
    log.info(
        "random late heads (50): share median %.4f, frac positive %.2f",
        out["random_late_heads"]["median_of_medians"],
        out["random_late_heads"]["median_frac_positive"],
    )
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
