from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("lens2an")


def band(ax, M, color, label, ls="-"):
    xs = np.arange(M.shape[1])
    ax.plot(xs, np.nanmedian(M, 0), color=color, lw=2, label=label, ls=ls)
    ax.fill_between(xs, np.nanpercentile(M, 25, 0), np.nanpercentile(M, 75, 0), color=color, alpha=0.15)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--lens2", type=Path, default=Path("work/lens2/lens2.npz"))
    parser.add_argument("--neurons", type=Path, default=Path("work/neurlens"))
    parser.add_argument("--figs", type=Path, default=Path("figures/lens2"))
    parser.add_argument("--band", default="19,22")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.figs.mkdir(parents=True, exist_ok=True)
    b0, b1 = (int(x) for x in args.band.split(","))
    R = np.load(args.lens2)
    B = R["cosE_m"].shape[1]
    fig, axes = plt.subplots(2, 3, figsize=(26, 12))
    ax = axes[0, 0]
    band(ax, R["cosE_a"], "#2563eb", "attention write · input form (E)")
    band(ax, R["cosU_a"], "#dc2626", "attention write · output form (U)")
    ax.set_title("attention writes: cosine with the marker-gap direction in embedding space vs unembedding space")
    ax = axes[0, 1]
    band(ax, R["cosE_m"], "#2563eb", "MLP write · input form (E)")
    band(ax, R["cosU_m"], "#dc2626", "MLP write · output form (U)")
    ax.set_title("MLP writes")
    ax = axes[0, 2]
    band(ax, R["cosE_v"][:, :B], "#2563eb", "v(L) · input form (E)")
    band(ax, R["cosU_v"][:, :B], "#dc2626", "v(L) · output form (U)")
    ax.set_title(f"running vector v(L); median cos(e_gap, u_gap) = {float(np.median(R['cos_eu'])):.3f}")
    ax = axes[1, 0]
    band(ax, R["gapE_m"], "#2563eb", "MLP write, embedding lens z-gap")
    band(ax, R["gapU_m"], "#dc2626", "MLP write, logit lens z-gap")
    band(ax, R["gapE_a"], "#93c5fd", "attention write, embedding lens", ls="--")
    band(ax, R["gapU_a"], "#fca5a5", "attention write, logit lens", ls="--")
    ax.set_title("z-scored marker gap under the two lenses")
    ax = axes[1, 1]
    band(ax, R["enE_m"], "#2563eb", "MLP write, energy in span E[markers]")
    band(ax, R["enU_m"], "#dc2626", "MLP write, energy in span U[markers]")
    band(ax, R["enE_a"], "#93c5fd", "attention write, span E", ls="--")
    band(ax, R["enU_a"], "#fca5a5", "attention write, span U", ls="--")
    ax.set_title("energy fraction of the unit write inside the ~60-dim marker spans (chance ≈ 0.015)")
    ax = axes[1, 2]
    if np.isfinite(R["resp_cosU"]).any():
        band(ax, R["resp_cosE"], "#2563eb", "MLP response to a push along v_k(b) · input form")
        band(ax, R["resp_cosU"], "#dc2626", "MLP response · output form")
        band(ax, R["resp_gain"], "#111827", "gain along v_k(b)", ls="--")
    ax.set_title("the MLP's response vector (stage G) through the two lenses")
    for ax in axes.flat:
        ax.axhline(0, color="k", lw=0.5)
        ax.axvspan(b0, b1, color="#f59e0b", alpha=0.12)
        ax.set_xlabel("block b")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.figs / "H1-forms.png", dpi=170)
    log.info("wrote %s", args.figs)
    for key in ("cosE_a", "cosU_a", "cosE_m", "cosU_m", "enE_m", "enU_m", "resp_cosU", "resp_cosE"):
        log.info("%-9s median by block: %s", key, [round(float(np.nanmedian(R[key][:, b])), 3) for b in range(0, B, 2)])
    N = json.loads((args.neurons / "neurons.json").read_text())
    for b in (2, 6, 10, 14, 18, 22, 26, 30, 34):
        rows = N.get(str(b), [])
        for r in rows[:2] + [x for x in rows if x["kind"] == "pos"][:2]:
            log.info(
                "block %d %s neuron %d (shared by %.0f%% of concepts, contrib %.4f, read_up*write %.3f, write %.3f) U+: %s | U-: %s",
                b,
                r["kind"],
                r["neuron"],
                100 * r["share_of_concepts"],
                r["mean_contrib"],
                r["read_up_x_write_mean"],
                r["write_mean"],
                " ".join(repr(t) for t in r["U_top"][:8]),
                " ".join(repr(t) for t in r["U_bottom"][:6]),
            )


if __name__ == "__main__":
    main()
