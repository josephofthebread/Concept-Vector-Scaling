import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

log = logging.getLogger("dose_analyze")
DANGER = ["tylenol", "syrup", "ibuprofen"]
CONTROL = "steps"


def rankdata(a):
    """Average ranks along axis 0 (rungs), vectorized over the remaining axis."""
    order = a.argsort(axis=0)
    ranks = np.empty_like(order, dtype=np.float64)
    n = a.shape[0]
    idx = np.arange(n)[:, None]
    np.put_along_axis(ranks, order, np.broadcast_to(idx, order.shape).astype(np.float64), axis=0)
    for col in range(a.shape[1]):
        _, inv, counts = np.unique(a[:, col], return_inverse=True, return_counts=True)
        cum = np.cumsum(counts)
        avg = cum - (counts - 1) / 2.0 - 1
        ranks[:, col] = avg[inv]
    return ranks


def contrast_matrix(npz, key, meta, L):
    """Danger-minus-steps Delta-cos contrast [layers, cols], plus per-ladder Spearman rho."""
    ##
    # NOTE: ranking uses the Delta-cos effect size, not Spearman rho: on nine rungs rho saturates
    # near +1 for random directions too, so rho is kept only as a monotonicity readout.
    ##
    lad_doses = {lad: meta["ladders"][lad]["doses"] for lad in meta["ladders"]}
    dcos, rho = {}, {}
    for lad in DANGER + [CONTROL]:
        arr = npz[f"{key}.{lad}"]
        dcos[lad] = arr[-1] - arr[0]
        rhos = []
        for li in range(L):
            cos = arr[:, li]
            x = np.log(np.asarray(lad_doses[lad], float))[:, None]
            xr = rankdata(np.broadcast_to(x, cos.shape).copy())[:, 0]
            yr = rankdata(cos)
            xr = xr - xr.mean()
            yr = yr - yr.mean(axis=0, keepdims=True)
            num = (xr[:, None] * yr).sum(axis=0)
            den = np.sqrt((xr**2).sum() * (yr**2).sum(axis=0)).clip(1e-12)
            rhos.append(num / den)
        rho[lad] = np.stack(rhos, axis=0)
    danger = np.mean([dcos[l] for l in DANGER], axis=0)
    return danger - dcos[CONTROL], rho


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    npz = np.load(args.readout, allow_pickle=True)
    meta = json.loads(str(npz["meta"]))
    layers = meta["layers"]
    names = meta["names"]
    L = len(layers)

    con_c, rho_c = contrast_matrix(npz, "cos", meta, L)
    con_x, _ = contrast_matrix(npz, "ctl", meta, L)
    null_mean = con_x.mean(axis=1, keepdims=True)
    null_std = con_x.std(axis=1, keepdims=True).clip(1e-9)
    z = (con_c - null_mean) / null_std

    best_li = z.argmax(axis=0)
    best_z = z.max(axis=0)
    dcos_tyl = npz["cos.tylenol"][-1] - npz["cos.tylenol"][0]
    order = np.argsort(-best_z)

    rows = []
    for c in order:
        li = int(best_li[c])
        rows.append(
            {
                "pair_id": int(c),
                "name": names[c],
                "z": round(float(best_z[c]), 2),
                "best_layer": int(layers[li]),
                "best_layer_idx": li,
                "danger_rho": round(float(np.mean([rho_c[l][li, c] for l in DANGER])), 3),
                "steps_rho": round(float(rho_c[CONTROL][li, c]), 3),
                "contrast": round(float(con_c[li, c]), 3),
                "dcos_tylenol": round(float(dcos_tyl[li, c]), 4),
            }
        )
    out = Path(args.readout).parent / f"ranking_{meta['tag']}.json"
    json.dump(
        {
            "model": meta["model"],
            "tag": meta["tag"],
            "layers": layers,
            "null_contrast_mean": [round(float(x), 3) for x in null_mean.ravel()],
            "null_contrast_std": [round(float(x), 3) for x in null_std.ravel()],
            "ranking": rows,
        },
        open(out, "w"),
        indent=1,
    )
    log.info("wrote %s (%d concepts)", out, len(rows))

    print(f"\n=== {meta['tag']} top {args.top} danger-dose concepts (z vs random null) ===")
    print(f"{'z':>6} {'pid':>4} {'layer':>5} {'dRho':>5} {'stpRho':>6} {'Δcos':>7}  concept")
    for r in rows[: args.top]:
        print(
            f"{r['z']:6.1f} {r['pair_id']:4d} {r['best_layer']:5d} {r['danger_rho']:5.2f} "
            f"{r['steps_rho']:6.2f} {r['dcos_tylenol']:7.4f}  {r['name']}"
        )
    print("\nlayer sweep (concepts with z>5 per layer):")
    for li, lay in enumerate(layers):
        print(f"  L{lay:2d}: {int((z[li] > 5).sum()):4d}  (max z {z[li].max():.1f})")


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--readout", required=True, help="results/dose_<tag>.npz")
    ap.add_argument("--top", type=int, default=25)
    main(ap.parse_args())
