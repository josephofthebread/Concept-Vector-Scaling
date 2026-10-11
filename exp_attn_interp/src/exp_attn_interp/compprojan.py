from __future__ import annotations

import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
from safetensors import safe_open

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("compprojan")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--compproj", type=Path, default=Path("work/compproj"))
    parser.add_argument("--out", type=Path, default=Path("work/compprojan.npz"))
    parser.add_argument("--figs", type=Path, default=Path("figures/probing"))
    parser.add_argument("--band", default="19,22")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.figs.mkdir(parents=True, exist_ok=True)
    b0, b1 = (int(x) for x in args.band.split(","))
    keys = ("a_mean", "a_std", "m_mean", "m_std", "e_mean", "e_std", "h_mean", "h_std", "cov_am")
    acc: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    pairs = []
    for f in sorted(args.compproj.glob("pair-*.safetensors")):
        with safe_open(str(f), framework="np") as h:
            for k in keys:
                acc[k].append(h.get_tensor(k))
            pairs.append(int(h.get_tensor("pair_id")))
    A = {k: np.stack(v) for k, v in acc.items()}
    K = len(pairs)
    B = A["a_mean"].shape[-1]
    gap_h = A["h_mean"][:, 0].mean(1) - A["h_mean"][:, 1].mean(1)
    noise_h = np.sqrt((A["h_std"] ** 2).mean((1, 2)))
    snr = gap_h / noise_h
    gap_a = A["a_mean"][:, 0].mean(1) - A["a_mean"][:, 1].mean(1)
    gap_m = A["m_mean"][:, 0].mean(1) - A["m_mean"][:, 1].mean(1)
    std_a = np.sqrt((A["a_std"] ** 2).mean((1, 2)))
    std_m = np.sqrt((A["m_std"] ** 2).mean((1, 2)))
    gap_e = A["e_mean"][:, 0].mean(1) - A["e_mean"][:, 1].mean(1)
    cov = A["cov_am"].mean((1, 2))
    spread = np.sqrt(0.5 * (A["h_mean"][:, 0].std(1) ** 2 + A["h_mean"][:, 1].std(1) ** 2))
    snr_story = gap_h / spread
    np.savez(
        str(args.out),
        pairs=np.array(pairs),
        gap_h=gap_h,
        noise_h=noise_h,
        snr=snr,
        gap_a=gap_a,
        gap_m=gap_m,
        std_a=std_a,
        std_m=std_m,
        gap_e=gap_e,
        cov=cov,
        snr_story=snr_story,
    )
    med = lambda x: [round(float(np.median(x[:, i])), 3) for i in range(0, x.shape[1], 2)]  # noqa: E731
    log.info("%d concepts. token SNR by row: %s (argmax median %d)", K, med(snr), int(np.median(np.argmax(snr, 1))))
    log.info("signal (pole gap) by row: %s", med(gap_h))
    log.info("token noise by row: %s", med(noise_h))
    log.info("story SNR by row: %s", med(snr_story))
    log.info("per-block attention gap: %s", med(gap_a))
    log.info("per-block MLP gap:       %s", med(gap_m))
    log.info("per-block attention token-std: %s", med(std_a))
    log.info("per-block MLP token-std:       %s", med(std_m))
    log.info("per-block attention gap/std: %s", med(gap_a / std_a))
    log.info("per-block MLP gap/std:       %s", med(gap_m / std_m))
    log.info(
        "cov(attention part, MLP part) of h_L·u across tokens, normalized by var(h_L·u): %s", med(cov / noise_h**2)
    )
    cumA = np.cumsum(gap_a, 1)
    cumM = np.cumsum(gap_m, 1)
    fig, axes = plt.subplots(2, 3, figsize=(26, 13))
    x = np.arange(B + 1)
    xb = np.arange(B)

    def band(ax, M, color, label, xs, ls="-"):
        ax.plot(xs, np.median(M, 0), color=color, lw=2, label=label, ls=ls)
        ax.fill_between(xs, np.percentile(M, 25, 0), np.percentile(M, 75, 0), color=color, alpha=0.15)

    ax = axes[0, 0]
    band(ax, snr, "#111827", "token-level SNR = pole gap / within-story token std", x)
    band(ax, snr_story / 3, "#7c3aed", "story-level SNR / 3", x, ls="--")
    ax.set_title("the readout's signal-to-noise by row (16 stories per pole)")
    ax = axes[0, 1]
    band(ax, gap_h, "#111827", "pole gap of h_L·u (signal)", x)
    band(ax, noise_h, "#dc2626", "token std of h_L·u (noise)", x)
    band(ax, cumA + gap_e[:, None], "#2563eb", "signal from attention (cumulative) + embedding", xb + 1, ls="--")
    band(ax, cumM, "#9ca3af", "signal from MLPs (cumulative)", xb + 1, ls="--")
    ax.set_title("signal and noise, and the signal's split by component")
    ax = axes[0, 2]
    band(ax, cov / noise_h**2, "#111827", "cov(attention part, MLP part) / var(h_L·u)", x)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("token-level covariance of the attention and MLP parts of the readout")
    ax = axes[1, 0]
    band(ax, gap_a, "#2563eb", "attention write: pole gap", xb)
    band(ax, gap_m, "#9ca3af", "MLP write: pole gap", xb)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("per-block signal (pole gap of the block's write along u)")
    ax = axes[1, 1]
    band(ax, std_a, "#2563eb", "attention write: token std", xb)
    band(ax, std_m, "#9ca3af", "MLP write: token std", xb)
    ax.set_title("per-block noise (within-story token std of the block's write along u)")
    ax = axes[1, 2]
    band(ax, gap_a / std_a, "#2563eb", "attention write: gap / token std", xb)
    band(ax, gap_m / std_m, "#9ca3af", "MLP write: gap / token std", xb)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("per-block signal-to-noise of each component's write")
    for ax in axes.flat:
        ax.axvspan(b0, b1, color="#f59e0b", alpha=0.12)
        ax.set_xlabel("row / block")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.figs / "P1-signal-noise.png", dpi=170)
    log.info("wrote %s", args.figs)


if __name__ == "__main__":
    main()
