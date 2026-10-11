from __future__ import annotations

import argparse
import glob
import json
import logging
import os

import numpy as np

log = logging.getLogger("genmono")
FEATURES = "../exp_lmsys_probe/data/features.json"
MINTURNS = 6


def per_episode_rho(z):
    """z [T, POS, L, P] -> rho [POS, L, P]: Spearman of each series vs turn index."""
    T = z.shape[0]
    k = np.arange(T, dtype=float)
    kc = k - k.mean()
    order = z.argsort(0)
    r = np.empty_like(order)
    np.put_along_axis(r, order, np.broadcast_to(np.arange(T)[:, None, None, None], z.shape), axis=0)
    rc = r.astype(float) - (T - 1) / 2.0
    den = np.sqrt((rc**2).sum(0) * (kc**2).sum())
    return np.where(den > 1e-9, (rc * kc[:, None, None, None]).sum(0) / den, 0.0)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="turnwork")
    ap.add_argument("--out", default="turnwork/genmono.json")
    ap.add_argument("--topk", type=int, default=30)
    args = ap.parse_args()
    feats = json.loads(open(FEATURES).read())["features"]
    feats.sort(key=lambda f: f["pair_id"])
    files = sorted(glob.glob(os.path.join(args.work, "*.npz")))
    e0 = np.load(files[0], allow_pickle=True)
    POS = [str(x) for x in e0["pos"]]
    LAYERS = [int(x) for x in e0["layers"]]
    P = e0["z"].shape[-1]

    acc = {
        g: {
            "sum": np.zeros((len(POS), len(LAYERS), P)),
            "sumsq": np.zeros((len(POS), len(LAYERS), P)),
            "sign": np.zeros((len(POS), len(LAYERS), P)),
            "n": 0,
        }
        for g in ("imp", "pos")
    }
    for f in files:
        d = np.load(f, allow_pickle=True)
        z = d["z"].astype(np.float32)
        if z.shape[0] < MINTURNS:
            continue
        g = "imp" if bool(d["impossible"]) else "pos"
        rho = per_episode_rho(z)
        acc[g]["sum"] += rho
        acc[g]["sumsq"] += rho**2
        acc[g]["sign"] += np.sign(rho)
        acc[g]["n"] += 1
    log.info("episodes with >=%d turns: imp=%d pos=%d", MINTURNS, acc["imp"]["n"], acc["pos"]["n"])

    def stats(g):
        n = acc[g]["n"]
        m = acc[g]["sum"] / n
        sd = np.sqrt(np.clip(acc[g]["sumsq"] / n - m**2, 0, None))
        se = sd / np.sqrt(n)
        cons = np.abs(acc[g]["sign"]) / n
        return m, se, cons

    m_i, se_i, cons_i = stats("imp")
    m_p, se_p, cons_p = stats("pos")

    result = {
        "minturns": MINTURNS,
        "n_imp": acc["imp"]["n"],
        "n_pos": acc["pos"]["n"],
        "positions": POS,
        "layers": LAYERS,
        "by_position": {},
    }
    for pi, pname in enumerate(POS):
        li = int(np.argmax([np.abs(m_i[pi, l]).mean() for l in range(len(LAYERS))]))
        mi, ci, zi = m_i[pi, li], cons_i[pi, li], np.abs(m_i[pi, li]) / (se_i[pi, li] + 1e-9)
        mp, cp = m_p[pi, li], cons_p[pi, li]
        score = ci * np.abs(mi)
        order = np.argsort(-score)[: args.topk]
        rows = [
            {
                "concept": feats[p]["concept"],
                "class": feats[p]["class_name"][:22],
                "mean_rho_imp": round(float(mi[p]), 3),
                "consistency_imp": round(float(ci[p]), 3),
                "z_across_ep": round(float(zi[p]), 1),
                "mean_rho_pos": round(float(mp[p]), 3),
                "consistency_pos": round(float(cp[p]), 3),
            }
            for p in order
        ]
        result["by_position"][pname] = {"layer": LAYERS[li], "top_genuinely_monotone": rows}
        top = rows[0]
        log.info(
            "[%s L%d] '%s' rho_imp=%.2f cons=%.2f z=%.0f (pos-side rho=%.2f cons=%.2f)",
            pname,
            LAYERS[li],
            top["concept"],
            top["mean_rho_imp"],
            top["consistency_imp"],
            top["z_across_ep"],
            top["mean_rho_pos"],
            top["consistency_pos"],
        )
    open(args.out, "w").write(json.dumps(result, indent=1))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
