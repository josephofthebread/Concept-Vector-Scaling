from __future__ import annotations

import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
from safetensors import safe_open

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("gensurv2an")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--gensurv2", type=Path, default=Path("work/gensurv2"))
    parser.add_argument("--figs", type=Path, default=Path("figures/dose"))
    parser.add_argument("--band", default="19,22")
    args = parser.parse_args()
    args.figs.mkdir(parents=True, exist_ok=True)
    b0, b1 = (int(x) for x in args.band.split(","))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    A, M, T, F, rows = [], [], [], [], None
    for f in sorted(args.gensurv2.glob("pair-*.safetensors")):
        with safe_open(str(f), framework="np") as h:
            A.append(h.get_tensor("attn"))
            M.append(h.get_tensor("mlp"))
            T.append(h.get_tensor("total"))
            F.append(h.get_tensor("full"))
            rows = h.get_tensor("rows").tolist()
    assert rows is not None
    Aa, Ma, Ta, Fa = (np.stack(x) for x in (A, M, T, F))
    K, R, P, B, H = Aa.shape
    attn_blk = np.median(Aa.mean(2).sum(-1), 0)
    mlp_blk = np.median(Ma.mean(2), 0)
    tot = np.median(Ta.mean(2), 0)
    full = np.median(Fa.mean(2), 0)
    log.info(
        "%d concepts, rows %s: context-only survival %s; full (context + current) %s",
        K,
        rows,
        [round(float(x), 2) for x in tot],
        [round(float(x), 2) for x in full],
    )
    for i, L in enumerate(rows):
        log.info(
            "row %d: attention transport by block (sum over heads) %s", L, [round(float(x), 3) for x in attn_blk[i][L:]]
        )
        log.info("row %d: MLP transport by block %s", L, [round(float(x), 3) for x in mlp_blk[i][L:]])
        log.info(
            "row %d: totals: attention %.2f, MLP %.2f; top heads %s",
            L,
            float(attn_blk[i].sum()),
            float(mlp_blk[i].sum()),
            [
                (int(b), int(h), round(float(v), 3))
                for b, h, v in sorted(
                    ((b, h, np.median(Aa[:, i, :, b, h].mean(1))) for b in range(B) for h in range(H)),
                    key=lambda x: -x[2],
                )[:6]
            ],
        )
    fig, axes = plt.subplots(1, 3, figsize=(26, 7))
    cmap = plt.get_cmap("viridis")
    for i, L in enumerate(rows):
        axes[0].plot(np.arange(B), attn_blk[i], marker=".", color=cmap(i / (R - 1)), lw=2, label=f"row {L}: attention")
        axes[1].plot(np.arange(B), mlp_blk[i], marker=".", color=cmap(i / (R - 1)), lw=2, label=f"row {L}: MLP")
    for ax, ttl in zip(
        axes[:2],
        (
            "attention writes carrying the injected context into the current token, by block",
            "MLP writes at the current token driven by the injected context, by block",
        ),
    ):
        ax.axhline(0, color="k", lw=0.5)
        ax.axvspan(b0, b1, color="#f59e0b", alpha=0.12)
        ax.set_xlabel("block")
        ax.set_title(ttl, fontsize=10)
        ax.legend(fontsize=8)
    ax = axes[2]
    w = 0.2
    for i, L in enumerate(rows):
        ax.bar(
            np.arange(3) + (i - 1.5) * w,
            [attn_blk[i].sum(), mlp_blk[i].sum(), tot[i]],
            width=w,
            color=cmap(i / (R - 1)),
            label=f"row {L}",
        )
    ax.set_xticks(range(3))
    ax.set_xticklabels(["attention total", "MLP total", "context-only survival"])
    ax.set_title("context transport by component (per unit |delta|, along v(35))")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.figs / "S6-context.png", dpi=170)
    log.info("wrote %s", args.figs)


if __name__ == "__main__":
    main()
