from __future__ import annotations

import argparse
import glob
import json
import logging
import os

import numpy as np

from exp_agentic.genmono import per_episode_rho

log = logging.getLogger("taskspec")
FEATURES = "../exp_lmsys_probe/data/features.json"
MINTURNS = 6


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="turnwork")
    ap.add_argument("--out", default="turnwork/taskspec.json")
    ap.add_argument("--topk", type=int, default=25)
    args = ap.parse_args()
    feats = json.loads(open(FEATURES).read())["features"]
    feats.sort(key=lambda f: f["pair_id"])
    files = sorted(glob.glob(os.path.join(args.work, "*.npz")))
    e0 = np.load(files[0], allow_pickle=True)
    POS = [str(x) for x in e0["pos"]]
    LAYERS = [int(x) for x in e0["layers"]]
    P = e0["z"].shape[-1]
    shp = (len(POS), len(LAYERS), P)
    acc = {g: {"s": np.zeros(shp), "ss": np.zeros(shp), "n": 0} for g in ("imp", "pos")}
    for f in files:
        d = np.load(f, allow_pickle=True)
        z = d["z"].astype(np.float32)
        if z.shape[0] < MINTURNS:
            continue
        g = "imp" if bool(d["impossible"]) else "pos"
        rho = per_episode_rho(z)
        acc[g]["s"] += rho
        acc[g]["ss"] += rho**2
        acc[g]["n"] += 1
    ni, npo = acc["imp"]["n"], acc["pos"]["n"]
    mi = acc["imp"]["s"] / ni
    sdi = np.sqrt(np.clip(acc["imp"]["ss"] / ni - mi**2, 0, None))
    mp = acc["pos"]["s"] / npo
    sdp = np.sqrt(np.clip(acc["pos"]["ss"] / npo - mp**2, 0, None))
    diff = mi - mp
    cohen = diff / np.sqrt((sdi**2 + sdp**2) / 2 + 1e-9)
    log.info("imp=%d pos=%d episodes (>=%d turns)", ni, npo, MINTURNS)

    result = {"n_imp": ni, "n_pos": npo, "positions": POS, "layers": LAYERS, "by_position": {}}
    for pi, pname in enumerate(POS):
        li = int(np.argmax([np.abs(cohen[pi, l]).mean() for l in range(len(LAYERS))]))
        c, di, dpo, dd = cohen[pi, li], mi[pi, li], mp[pi, li], diff[pi, li]

        def rows(order):
            return [
                {
                    "concept": feats[p]["concept"],
                    "class": feats[p]["class_name"][:22],
                    "rho_imp": round(float(di[p]), 3),
                    "rho_pos": round(float(dpo[p]), 3),
                    "diff": round(float(dd[p]), 3),
                    "cohen_d": round(float(c[p]), 2),
                }
                for p in order
            ]

        result["by_position"][pname] = {
            "layer": LAYERS[li],
            "impossible_specific": rows(np.argsort(c)[: args.topk]),
            "possible_specific": rows(np.argsort(-c)[: args.topk]),
        }
        top = result["by_position"][pname]["impossible_specific"][0]
        log.info(
            "[%s L%d] most impossible-specific: '%s' rho_imp=%.2f rho_pos=%.2f d=%.2f",
            pname,
            LAYERS[li],
            top["concept"],
            top["rho_imp"],
            top["rho_pos"],
            top["cohen_d"],
        )
    open(args.out, "w").write(json.dumps(result, indent=1))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
