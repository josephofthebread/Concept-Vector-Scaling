from __future__ import annotations

import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
from safetensors import safe_open

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("injrows")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--rows", default="12,16,20,24,28")
    parser.add_argument("--figs", type=Path, default=Path("figures/formation"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = [int(x) for x in args.rows.split(",")]
    fig, axes = plt.subplots(1, 3, figsize=(26, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(rows)))
    for L, color in zip(rows, colors):
        d = Path(f"work/inject_L{L}")
        files = sorted(d.glob("pair-*.safetensors"))
        if not files:
            continue
        mlp_L, att_L = [], []
        for f in files:
            with safe_open(str(f), framework="np") as h:
                mlp_L.append(h.get_tensor("mlp_proj_L"))
                att_L.append(h.get_tensor("head_proj_L").sum(-1))
        mL, aL = np.median(np.stack(mlp_L), 0), np.median(np.stack(att_L), 0)
        xs = np.arange(L, L + len(mL))
        axes[0].plot(xs, mL, "-", color=color, lw=2, label=f"injected at row {L}")
        axes[1].plot(xs, aL, "-", color=color, lw=2, label=f"injected at row {L}")
        axes[2].plot(
            xs, 1 + np.cumsum(mL + aL), "-", color=color, lw=2, label=f"row {L}: carry of the injected direction"
        )
        log.info(
            "row %d (%d concepts): MLP transport along the injected direction, first 6 blocks %s; blocks 22+ mean %.3f; attention first 6 %s",
            L,
            len(files),
            [round(float(x), 3) for x in mL[:6]],
            float(mL[max(0, 22 - L) :].mean()) if L < 22 else float(mL.mean()),
            [round(float(x), 3) for x in aL[:6]],
        )
    axes[0].set_title("MLP write change along the injected direction, per block (/ |delta|)")
    axes[1].set_title("attention write change along the injected direction, per block (/ |delta|)")
    axes[2].set_title("cumulative: how much of the injected direction is present after each block")
    for ax in axes:
        ax.axhline(0 if ax is not axes[2] else 1, color="k", lw=0.5)
        ax.axvspan(19, 22, color="#f59e0b", alpha=0.12)
        ax.set_xlabel("block")
        ax.legend(fontsize=8)
    fig.tight_layout()
    args.figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figs / "A7-inject-rows.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
