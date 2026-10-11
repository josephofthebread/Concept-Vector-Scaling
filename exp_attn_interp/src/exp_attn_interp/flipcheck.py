from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from exp_attn_interp.amp import np_ranks  # noqa: E402

log = logging.getLogger("flipcheck")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--curves", type=Path, default=Path("work/formation/curves.npz"))
    parser.add_argument("--pairs", type=Path, default=Path("work/formation/pairs.npy"))
    parser.add_argument("--figs", type=Path, default=Path("figures/formation"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    C = np.load(args.curves)
    pairs = np.load(args.pairs)
    sbl = json.loads(Path("work/steer_by_layer.json").read_text())["qwen"]
    steer = json.loads(Path("work/steerability.json").read_text())["qwen"]
    r = C["along_m"] / np.sqrt(C["along_m"] ** 2 + C["perp_m"] ** 2 + 1e-12)
    K, B = r.shape
    ker = np.ones(3) / 3
    rs = np.stack([np.convolve(row, ker, mode="same") for row in r])
    flip = np.full(K, -1)
    for i in range(K):
        for b in range(8, B - 2):
            if rs[i, b] > 0 and rs[i, b + 1] > 0 and rs[i, b + 2] > 0:
                flip[i] = b
                break
    ra = C["along_a"] / np.sqrt(C["along_a"] ** 2 + C["perp_a"] ** 2 + 1e-12)
    peak_a = np.argmax(C["along_a"][:, 10:32], 1) + 10
    half = np.array([int(np.argmax(C["to_last"][i] >= 0.5)) for i in range(K)])
    best_row = np.full(K, np.nan)
    score_opt = np.full(K, np.nan)
    for i, k in enumerate(pairs):
        d = sbl["score"].get(str(int(k)))
        if d:
            rows_ = sorted(int(x) for x in d)
            best_row[i] = rows_[int(np.argmax([d[str(x)] for x in rows_]))]
            score_opt[i] = d.get("21", np.nan)
    st = np.array([steer.get(str(int(k)), np.nan) for k in pairs])
    ok = (flip >= 0) & ~np.isnan(best_row)
    log.info(
        "flip block: median %d (p10 %d, p90 %d); no flip for %d concepts",
        int(np.median(flip[flip >= 0])),
        int(np.percentile(flip[flip >= 0], 10)),
        int(np.percentile(flip[flip >= 0], 90)),
        int((flip < 0).sum()),
    )
    sp = lambda a, b: float(np.corrcoef(np_ranks(a), np_ranks(b))[0, 1])  # noqa: E731
    log.info(
        "per-concept best steering row: median %d; rho(flip block, best row) %+.3f; rho(attention-peak block, best row) %+.3f; "
        "rho(half-alignment layer, best row) %+.3f",
        int(np.nanmedian(best_row)),
        sp(flip[ok], best_row[ok]),
        sp(peak_a[ok], best_row[ok]),
        sp(half[ok], best_row[ok]),
    )
    okk = (flip >= 0) & ~np.isnan(st)
    log.info(
        "rho(flip block, judged steerability) %+.3f; rho(min over 8..19 of MLP cos, steerability) %+.3f; "
        "rho(mean MLP cos over 22..34, steerability) %+.3f; rho(mean attention cos over 19..34, steerability) %+.3f",
        sp(flip[okk], st[okk]),
        sp(rs[:, 8:20].min(1)[okk], st[okk]),
        sp(rs[:, 22:35].mean(1)[okk], st[okk]),
        sp(ra[:, 19:35].mean(1)[okk], st[okk]),
    )
    fig, axes = plt.subplots(1, 3, figsize=(24, 6.5))
    axes[0].hist(flip[flip >= 0], bins=np.arange(8, 36) - 0.5, color="#9ca3af")
    axes[0].axvspan(19, 22, color="#f59e0b", alpha=0.25, label="judged steering band")
    axes[0].set_title("block where the MLP write turns from opposing to reinforcing v̂(b), per concept")
    axes[0].set_xlabel("block")
    axes[0].legend()
    terc = np.array_split(np.argsort(np.nan_to_num(st, nan=-1)), 3)
    for t, color, lab in zip(
        terc, ("#dc2626", "#9ca3af", "#2563eb"), ("least steerable third", "middle", "most steerable third")
    ):
        axes[1].plot(range(B), np.median(rs[t], 0), color=color, lw=2, label=lab)
        axes[2].plot(range(B), np.median(ra[t], 0), color=color, lw=2, label=lab)
    for ax, ttl in zip(
        axes[1:], ("cos(MLP write, v̂(b)) by steerability tercile", "cos(attention write, v̂(b)) by steerability tercile")
    ):
        ax.axhline(0, color="k", lw=0.5)
        ax.axvspan(19, 22, color="#f59e0b", alpha=0.12)
        ax.set_title(ttl)
        ax.set_xlabel("block")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.figs / "A5-flip.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
