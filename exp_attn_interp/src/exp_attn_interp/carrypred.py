from __future__ import annotations

import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
from safetensors import safe_open

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("carrypred")
ROWS = (12, 16, 20, 24, 28)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--mlpgain", type=Path, default=Path("work/mlpgain"))
    parser.add_argument("--inject", default="work/inject_L{L}")
    parser.add_argument("--figs", type=Path, default=Path("figures/gain"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.figs.mkdir(parents=True, exist_ok=True)
    gains = {}
    for f in sorted(args.mlpgain.glob("pair-*.safetensors")):
        with safe_open(str(f), framework="np") as h:
            g = h.get_tensor("gain")
        gains[int(f.stem.split("-")[1])] = g
    B = next(iter(gains.values())).shape[1]
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    rows_first, rows_carry = [], []
    for L in ROWS:
        d = Path(args.inject.format(L=L))
        xs, ys, cs_meas, cs_pred, cs_pred_n = [], [], [], [], []
        for f in sorted(d.glob("pair-*.safetensors")):
            k = int(f.stem.split("-")[1])
            if k not in gains:
                continue
            with safe_open(str(f), framework="np") as h:
                mp = h.get_tensor("mlp_proj_L")
                carry = float(h.get_tensor("carry"))
            g_st = gains[k][:2, :, 0, 1].mean(0)
            g_ne = gains[k][2, :, 0, 1]
            xs.append(float(g_ne[L]))
            ys.append(float(mp[0]))
            cs_meas.append(carry)
            cs_pred.append(float(np.prod(1 + g_st[L : B - 1])))
            cs_pred_n.append(float(np.prod(1 + g_ne[L : B - 1])))
        xa, ya, cm_, cp_, cpn_ = (np.array(v) for v in (xs, ys, cs_meas, cs_pred, cs_pred_n))
        r1 = np.corrcoef(xa, ya)[0, 1]
        log.info(
            "row %d (%d concepts): first-block MLP transport measured median %.3f vs gain (neutral, eps 0.25) %.3f, r %.2f | carry measured %.2f vs predicted from gains (neutral) %.2f / (stories) %.2f",
            L,
            len(xa),
            np.median(ya),
            np.median(xa),
            r1,
            np.median(cm_),
            np.median(cpn_),
            np.median(cp_),
        )
        rows_first.append((L, np.median(ya), np.median(xa)))
        rows_carry.append((L, np.median(cm_), np.median(cpn_), np.median(cp_)))
        axes[0].scatter(xa, ya, s=8, alpha=0.6, label=f"row {L} (r {r1:.2f})")
    axes[0].plot([-0.3, 0.2], [-0.3, 0.2], color="k", lw=0.5)
    axes[0].set_xlabel("MLP gain along v_k(L) at block L, eps 0.25, neutral prompts (gain stage)")
    axes[0].set_ylabel("measured MLP transport of the injection at block L (injection stage)")
    axes[0].set_title("prediction (1): the first MLP's response to the injection")
    axes[0].legend(fontsize=8)
    Ls = [r[0] for r in rows_carry]
    axes[1].plot(
        Ls, [r[1] for r in rows_carry], marker="o", color="#111827", lw=2, label="measured carry (injection stage)"
    )
    axes[1].plot(
        Ls,
        [r[2] for r in rows_carry],
        marker="o",
        color="#dc2626",
        lw=2,
        label="predicted: prod (1 + gain_b), neutral prompts",
    )
    axes[1].plot(
        Ls, [r[3] for r in rows_carry], marker="o", color="#2563eb", lw=2, ls="--", label="predicted, gains on stories"
    )
    axes[1].set_xlabel("injection row L")
    axes[1].set_title("prediction (2): the carry of the injected direction to the last row")
    axes[1].legend(fontsize=8)
    axes[2].plot(
        Ls, [r[1] for r in rows_first], marker="o", color="#111827", lw=2, label="measured first-block MLP transport"
    )
    axes[2].plot(Ls, [r[2] for r in rows_first], marker="o", color="#dc2626", lw=2, label="gain at block L (eps 0.25)")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_xlabel("injection row L")
    axes[2].set_title("medians of prediction (1) by row")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.figs / "G2-carry.png", dpi=170)
    log.info("wrote %s", args.figs)


if __name__ == "__main__":
    main()
