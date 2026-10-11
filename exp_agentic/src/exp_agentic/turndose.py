from __future__ import annotations

import argparse
import glob
import json
import logging
import os

import numpy as np

log = logging.getLogger("turndose")

FEATURES = "../exp_lmsys_probe/data/features.json"
WATCH = [
    "shortcut acceptance",
    "reward hacking",
    "cheating",
    "deception",
    "gaming the metric",
    "cutting corners",
    "giving up",
    "learned helplessness",
    "persistence",
    "self-preservation",
    "honest failure",
    "sunk cost fallacy",
]
KMAX, MINEP = 14, 5


def rankdata(a):
    o = a.argsort()
    r = np.empty(len(a))
    r[o] = np.arange(len(a))
    return r


def spearman(y):
    """Spearman of a per-turn curve y[k] vs turn index k (over the valid entries)."""
    k = np.arange(len(y), dtype=float)
    if len(y) < 3 or np.allclose(y, y[0]):
        return 0.0
    rx, ry = rankdata(k), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 1e-9 else 0.0


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="turnwork")
    ap.add_argument("--out", default="turnwork/turndose.json")
    ap.add_argument("--topk", type=int, default=25)
    args = ap.parse_args()

    feats = json.loads(open(FEATURES).read())["features"]
    feats.sort(key=lambda f: f["pair_id"])
    name2p = {f["concept"]: f["pair_id"] for f in feats}
    files = sorted(glob.glob(os.path.join(args.work, "*.npz")))
    e0 = np.load(files[0], allow_pickle=True)
    POS = [str(x) for x in e0["pos"]]
    LAYERS = [int(x) for x in e0["layers"]]
    P = e0["z"].shape[-1]
    log.info("%d episodes; POS=%s LAYERS=%s P=%d", len(files), POS, LAYERS, P)

    shape = (KMAX, len(POS), len(LAYERS), P)
    S = {g: np.zeros(shape) for g in ("imp", "pos")}
    C = {g: np.zeros(KMAX) for g in ("imp", "pos")}
    ep_rows = []
    n_imp = n_pos = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        z = d["z"].astype(np.float32)
        T = z.shape[0]
        g = "imp" if bool(d["impossible"]) else "pos"
        kk = min(T, KMAX)
        S[g][:kk] += z[:kk]
        C[g][:kk] += 1
        ep_rows.append((g, str(d["ending"]), z.mean(0)))
        n_imp += g == "imp"
        n_pos += g == "pos"
    log.info("impossible episodes=%d, possible=%d", n_imp, n_pos)

    cells = {
        g: np.divide(S[g], C[g][:, None, None, None], out=np.zeros(shape), where=C[g][:, None, None, None] > 0)
        for g in ("imp", "pos")
    }
    valid = {g: C[g] >= MINEP for g in ("imp", "pos")}

    def rho_field(g):
        vk = valid[g]
        out = np.zeros((len(POS), len(LAYERS), P))
        for pi in range(len(POS)):
            for li in range(len(LAYERS)):
                curve = cells[g][vk, pi, li, :]
                for p in range(P):
                    out[pi, li, p] = spearman(curve[:, p])
        return out

    R_imp, R_pos = rho_field("imp"), rho_field("pos")

    result = {
        "n_impossible": n_imp,
        "n_possible": n_pos,
        "positions": POS,
        "layers": LAYERS,
        "kmax": KMAX,
        "minep": MINEP,
        "by_position": {},
        "watchlist": {},
    }

    for pi, pname in enumerate(POS):
        li = int(np.argmax([np.abs(R_imp[pi, l]).mean() for l in range(len(LAYERS))]))
        ri, rp = R_imp[pi, li], R_pos[pi, li]
        diff = np.abs(ri) - np.abs(rp)

        def rows(order):
            return [
                {
                    "concept": feats[p]["concept"],
                    "class": feats[p]["class_name"][:24],
                    "rho_imp": round(float(ri[p]), 3),
                    "rho_pos": round(float(rp[p]), 3),
                    "diff": round(float(diff[p]), 3),
                }
                for p in order
            ]

        result["by_position"][pname] = {
            "layer": LAYERS[li],
            "top_monotonic_impossible": rows(np.argsort(-np.abs(ri))[: args.topk]),
            "top_differential": rows(np.argsort(-diff)[: args.topk]),
        }
        log.info(
            "[%s L%d] top-mono '%s' rho_imp=%.2f | top-diff '%s' diff=%.2f",
            pname,
            LAYERS[li],
            feats[int(np.argsort(-np.abs(ri))[0])]["concept"],
            ri[np.argsort(-np.abs(ri))[0]],
            feats[int(np.argsort(-diff)[0])]["concept"],
            diff[np.argsort(-diff)[0]],
        )

    for c in WATCH:
        if c not in name2p:
            continue
        p = name2p[c]
        best = max(
            ((pi, li) for pi in range(len(POS)) for li in range(len(LAYERS))), key=lambda t: abs(R_imp[t[0], t[1], p])
        )
        pi, li = best
        result["watchlist"][c] = {
            "position": POS[pi],
            "layer": LAYERS[li],
            "rho_imp": round(float(R_imp[pi, li, p]), 3),
            "rho_pos": round(float(R_pos[pi, li, p]), 3),
        }

    flatdiff = np.abs(R_imp) - np.abs(R_pos)
    pi, li, p = np.unravel_index(int(flatdiff.argmax()), flatdiff.shape)
    hack = [r[2][pi, li, p] for r in ep_rows if r[0] == "imp" and r[1] == "submit"]
    giveup = [r[2][pi, li, p] for r in ep_rows if r[0] == "imp" and r[1] == "give_up"]
    result["ending_breakdown"] = {
        "concept": feats[int(p)]["concept"],
        "position": POS[int(pi)],
        "layer": LAYERS[int(li)],
        "mean_z_submit_hack": round(float(np.mean(hack)), 3) if hack else None,
        "n_submit": len(hack),
        "mean_z_give_up": round(float(np.mean(giveup)), 3) if giveup else None,
        "n_give_up": len(giveup),
    }

    open(args.out, "w").write(json.dumps(result, indent=1))
    log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
