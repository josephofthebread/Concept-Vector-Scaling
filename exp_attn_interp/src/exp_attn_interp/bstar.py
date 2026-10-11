from __future__ import annotations

import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
from safetensors import safe_open

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("bstar")


def crossing(c: np.ndarray, lo: int = 4) -> int:
    """Sign rule: first block whose write is positive for three blocks after a non-positive one."""
    B = len(c)
    for b in range(lo, B - 2):
        if c[b] > 0 and c[b + 1] > 0 and c[b + 2] > 0 and (c[max(0, b - 4) : b] <= 0).any():
            return b
    return -1


def crossing_cum(c: np.ndarray) -> int:
    """Cumulative rule: the block after the MLP stack's cumulative write reaches its minimum."""
    return int(np.argmin(np.cumsum(c[:-1]))) + 1


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--decomp", type=Path, default=Path("work/decomp"))
    parser.add_argument("--curves", type=Path, default=Path("work/formation/curves.npz"))
    parser.add_argument("--out", type=Path, default=Path("work/bstar"))
    parser.add_argument("--figs", type=Path, default=Path("figures/formation"))
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    args.figs.mkdir(parents=True, exist_ok=True)
    files = sorted(args.decomp.glob("item-*.safetensors"))
    per: dict[int, dict[tuple[int, int], np.ndarray]] = {}
    read_layers = None
    for f in files:
        with safe_open(str(f), framework="np") as h:
            cm = h.get_tensor("c_mlp")
            k = int(h.get_tensor("pair_id"))
            side = int(h.get_tensor("side"))
            if read_layers is None:
                read_layers = [int(x) for x in h.metadata()["read_layers"].split(",")]
        item = int(f.stem.split("-")[1]) - 200000
        story = (item - k * 20) // 2
        per.setdefault(k, {})[(story, side)] = cm
    assert read_layers is not None
    pairs = sorted(k for k in per if len(per[k]) == 8)
    nL, B = per[pairs[0]][(0, 0)].shape
    S = np.stack([np.stack([np.stack([per[k][(s, side)] for side in (0, 1)]) for s in range(4)]) for k in pairs])
    log.info("%d concepts with 4+4 stories; read layers %s", len(pairs), read_layers)
    D = S[:, :, 0] - S[:, :, 1]
    last = nL - 1
    full = np.median(D[:, :, last].mean(1), 0)
    bstar = crossing(full)
    log.info(
        "population curve along v(last), 4 stories/pole, all concepts: b* = %d; curve %s",
        bstar,
        [round(float(x), 3) for x in full],
    )
    log.info("cumulative rule on the population curve: b* = %d", crossing_cum(full))
    per_k = np.array([crossing(D[i, :, last].mean(0)) for i in range(len(pairs))])
    per_kc = np.array([crossing_cum(D[i, :, last].mean(0)) for i in range(len(pairs))])
    log.info(
        "per-concept b*_k (sign rule): median %.0f, p10 %.0f, p90 %.0f, undefined %d",
        np.median(per_k[per_k >= 0]),
        np.percentile(per_k[per_k >= 0], 10),
        np.percentile(per_k[per_k >= 0], 90),
        int((per_k < 0).sum()),
    )
    log.info(
        "per-concept b*_k (cumulative rule): median %.0f, p10 %.0f, p90 %.0f; histogram %s",
        np.median(per_kc),
        np.percentile(per_kc, 10),
        np.percentile(per_kc, 90),
        np.bincount(per_kc, minlength=B).tolist(),
    )
    by_row = {
        read_layers[i]: (crossing(np.median(D[:, :, i].mean(1), 0)), crossing_cum(np.median(D[:, :, i].mean(1), 0)))
        for i in range(nL)
    }
    log.info("b* (sign rule, cumulative rule) by the row of the direction used: %s", by_row)
    if args.curves.exists():
        cv = np.load(args.curves)
        am = cv["along_m"]
        am = am if am.ndim == 2 else am[None]
        cur_r = np.median(am, 0)
        per_r = np.array([crossing_cum(am[i]) for i in range(am.shape[0])])
        log.info(
            "along the RUNNING direction v(b) (vdecomp, %d concepts): population b* sign %d / cumulative %d; per-concept cumulative median %.0f p10 %.0f p90 %.0f",
            am.shape[0],
            crossing(cur_r),
            crossing_cum(cur_r),
            np.median(per_r),
            np.percentile(per_r, 10),
            np.percentile(per_r, 90),
        )
    rng = np.random.default_rng(args.seed)
    ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, len(pairs)]
    ns = [1, 2, 4]
    est = np.full((len(ns), len(ms), args.draws), -1)
    estc = np.full((len(ns), len(ms), args.draws), -1)
    for a, n in enumerate(ns):
        for c, m in enumerate(ms):
            for d in range(args.draws):
                ks = rng.choice(len(pairs), size=m, replace=False)
                ss = rng.choice(4, size=n, replace=False)
                curve = np.median(D[ks][:, ss, last].mean(1), 0)
                est[a, c, d] = crossing(curve)
                estc[a, c, d] = crossing_cum(curve)
    np.savez(
        str(args.out / "bstar.npz"),
        full=full,
        per_k=per_k,
        per_kc=per_kc,
        est=est,
        estc=estc,
        ms=np.array(ms),
        ns=np.array(ns),
        pairs=np.array(pairs),
    )
    for a, n in enumerate(ns):
        log.info(
            "%d stories/pole, sign rule: b* median [p10,p90] by #concepts %s: %s",
            n,
            ms,
            [
                f"{int(np.median(est[a, c]))} [{int(np.percentile(est[a, c], 10))},{int(np.percentile(est[a, c], 90))}]"
                for c in range(len(ms))
            ],
        )
        log.info(
            "%d stories/pole, cumulative rule: %s",
            n,
            [
                f"{int(np.median(estc[a, c]))} [{int(np.percentile(estc[a, c], 10))},{int(np.percentile(estc[a, c], 90))}]"
                for c in range(len(ms))
            ],
        )
    running = None
    if args.curves.exists():
        cv = np.load(args.curves)
        if "along_m" in cv.files or "pole_m" in cv.files:
            key = [x for x in cv.files if x.startswith("along") and "m" in x]
            log.info("curves keys: %s", cv.files)
            running = key
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    x = np.arange(B)
    ax = axes[0]
    for i, L in enumerate(read_layers):
        if L < 24:
            continue
        cur = np.median(D[:, :, i].mean(1), 0)
        ax.plot(
            x,
            cur / np.abs(cur).max(),
            lw=2 if L == read_layers[-1] else 1,
            label=f"along the paper's v(row {L}): b* = {crossing_cum(cur)}",
        )
    if args.curves.exists():
        cur = np.median(np.load(args.curves)["along_m"], 0)
        ax.plot(
            np.arange(len(cur)),
            cur / np.abs(cur).max(),
            lw=2,
            color="#dc2626",
            label=f"along the running direction v(b): b* = {crossing_cum(cur)}",
        )
    ax.axhline(0, color="k", lw=0.5)
    ax.axvspan(19, 22, color="#f59e0b", alpha=0.12)
    ax.set_title(
        "population curve c(b): median over concepts of the per-block MLP write along the direction (normalized)"
    )
    ax.set_xlabel("block b")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.hist(per_kc, bins=np.arange(-0.5, B + 0.5, 1), color="#7c3aed", alpha=0.8, label="cumulative rule")
    ax.hist(per_k[per_k >= 0], bins=np.arange(-0.5, B + 0.5, 1), color="#9ca3af", alpha=0.5, label="sign rule")
    ax.set_title(f"per-concept crossing b*_k (4 stories/pole); cumulative-rule median {np.median(per_kc):.0f}")
    ax.set_xlabel("b*_k")
    ax.legend(fontsize=8)
    ax = axes[2]
    for a, n in enumerate(ns):
        med = [np.median(estc[a, c]) for c in range(len(ms))]
        lo = [np.percentile(estc[a, c], 10) for c in range(len(ms))]
        hi = [np.percentile(estc[a, c], 90) for c in range(len(ms))]
        ax.plot(ms, med, marker="o", label=f"{n} stories per pole (cumulative rule)")
        ax.fill_between(ms, lo, hi, alpha=0.15)
    ax.set_xscale("log")
    ax.axhline(bstar, color="k", lw=0.5, ls="--")
    ax.set_title("b* estimated from m concepts (median, p10–p90 over random subsets)")
    ax.set_xlabel("concepts m")
    ax.set_ylabel("b*")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.figs / "A10-bstar.png", dpi=170)
    log.info("wrote %s (running-direction keys: %s)", args.figs, running)


if __name__ == "__main__":
    main()
