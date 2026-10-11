from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
from safetensors import safe_open

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("bandfig")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--figs", type=Path, default=Path("figures/formation"))
    parser.add_argument("--band", default="19,22")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    b0, b1 = (int(x) for x in args.band.split(","))
    G = np.load("work/gainan/gainan.npz")
    gain = np.median(G["mid"][:, :2, :, 0].mean(1), 0)
    gate = np.median(G["path"][:, :2, :, 0, 1].mean(1), 0)
    up = np.median(G["path"][:, :2, :, 0, 0].mean(1), 0)
    B = gain.shape[0]
    E = json.loads(Path("work/steereff-qwen.json").read_text())
    c = {int(k): v for k, v in E["efficacy"]["concept pole"].items()}
    S, C, rows = [], [], None
    for f in sorted(Path("work/gensurv").glob("pair-*.safetensors")):
        with safe_open(str(f), framework="np") as h:
            S.append(h.get_tensor("surv"))
            C.append(h.get_tensor("ctx"))
            rows = h.get_tensor("rows").tolist()
    assert rows is not None
    Sa, Ca = np.stack(S), np.stack(C)
    surv0 = np.median(Sa[:, :, :, 0].mean(2), 0)
    survm = np.median(Sa.mean((2, 3)), 0)
    ctx15 = np.median(Ca[:, :, :, 15].mean(2), 0)
    P = np.load("work/compprojan.npz")
    snr = np.median(P["snr"], 0)
    cov = np.median(P["cov"] / P["noise_h"] ** 2, 0)
    inv = Path("work/langprof.npz")
    xp = Path("work/langxprobe.npz")
    n_panels = 4 if inv.exists() else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 5 * n_panels), sharex=True)
    x = np.arange(B)
    ax = axes[0]
    ax.plot(x, gain, color="#111827", lw=2.5, label="MLP gain along the concept direction (push 0.25)")
    ax.plot(x, up, color="#2563eb", lw=1.5, ls="--", label="up path (closed-neuron contraction)")
    ax.plot(x, gate, color="#dc2626", lw=1.5, ls="--", label="gate path (concept-keyed neurons)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("gain per block")
    ax.set_title("the MLP stack: contraction below, concept-keyed amplification from 22")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(
        sorted(c),
        [c[L] for L in sorted(c)],
        marker="s",
        color="#dc2626",
        lw=2.5,
        label="judged steering efficacy per unit push, c_L (c_22 = 1)",
    )
    ax.plot(
        rows,
        surv0 / surv0[rows.index(22)],
        marker="o",
        color="#9ca3af",
        lw=1.5,
        label="current-token survival of an injection (÷ row 22)",
    )
    ax.plot(
        rows,
        survm / survm[rows.index(22)],
        marker="o",
        color="#111827",
        lw=1.5,
        label="survival averaged over a 24-token generation (÷ row 22)",
    )
    ax.plot(
        rows,
        ctx15 / ctx15[rows.index(22)],
        marker="o",
        color="#2563eb",
        lw=1.5,
        ls="--",
        label="context-only survival at generated token 15 (÷ row 22)",
    )
    ax.set_ylabel("relative to row 22")
    ax.set_title("steering: the rise is the carried amplitude, the fall is the context path")
    ax.legend(fontsize=8)
    ax = axes[2]
    ax.plot(np.arange(B + 1), snr, color="#111827", lw=2.5, label="token-level SNR of the readout")
    ax2 = ax.twinx()
    ax2.plot(np.arange(B + 1), cov, color="#7c3aed", lw=1.5, ls="--", label="attention–MLP token covariance / variance")
    ax2.axhline(0, color="#7c3aed", lw=0.5)
    ax2.set_ylim(-0.7, 0.7)
    ax2.set_ylabel("covariance share", color="#7c3aed")
    ax.set_ylabel("SNR")
    if n_panels == 3:
        ax.set_xlabel("row / block")
    ax.set_title("probing: SNR rises while the MLPs cancel attention's token noise, plateaus when they multiply it")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
    if n_panels == 4:
        ax = axes[3]
        inv_data = np.load(str(inv))
        xr = np.arange(inv_data["res"].shape[1])
        ax.plot(
            xr,
            np.median(inv_data["res"], 0),
            color="#dc2626",
            lw=2.5,
            label="cross-language cosine of the concept-specific residual (5 languages)",
        )
        ax.plot(
            xr, np.median(inv_data["st"], 0), color="#2563eb", lw=1.5, label="cross-language cosine of the stance part"
        )
        ax.plot(
            xr,
            np.median(inv_data["rshare"], 0),
            color="#111827",
            lw=1.5,
            ls="--",
            label="share of the residual in the vector's energy",
        )
        if xp.exists():
            X = np.load(str(xp))
            s = np.median(X["snr"][:, 0, 1:, :].mean(1), 0)
            ax.plot(
                np.arange(len(s)),
                s / s.max(),
                color="#16a34a",
                lw=1.5,
                ls=":",
                label="English-built vector as a probe on the other four languages (story SNR ÷ max)",
            )
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("cosine / share")
        ax.set_xlabel("row / block")
        ax.set_title("invariance: the residual is language-free only in the band; the stance part everywhere")
        ax.legend(fontsize=8, loc="lower center")
    for a in axes:
        a.axvspan(b0, b1, color="#f59e0b", alpha=0.15)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.figs / ("A14-band.png" if n_panels == 4 else "A13-band.png"), dpi=170)
    log.info("wrote %s", args.figs)


if __name__ == "__main__":
    main()
