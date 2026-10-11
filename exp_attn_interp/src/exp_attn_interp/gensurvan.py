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

log = logging.getLogger("gensurvan")
CTX_T = (0, 3, 7, 15, 23)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--gensurv", type=Path, default=Path("work/gensurv"))
    parser.add_argument("--steereff", type=Path, default=Path("work/steereff-qwen.json"))
    parser.add_argument("--figs", type=Path, default=Path("figures/dose"))
    parser.add_argument("--ref-row", type=int, default=22)
    parser.add_argument("--band", default="19,22")
    args = parser.parse_args()
    ref = args.ref_row
    b0, b1 = (int(x) for x in args.band.split(","))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.figs.mkdir(parents=True, exist_ok=True)
    S, C, G, rows = [], [], [], None
    for f in sorted(args.gensurv.glob("pair-*.safetensors")):
        with safe_open(str(f), framework="np") as h:
            S.append(h.get_tensor("surv"))
            C.append(h.get_tensor("ctx"))
            G.append(h.get_tensor("gain"))
            rows = h.get_tensor("rows").tolist()
    assert rows is not None
    Sa, Ca, Ga = (np.stack(x) for x in (S, C, G))
    K, R, P, N = Sa.shape
    surv_t = np.median(Sa.mean(2), 0)
    surv_mean = np.median(Sa.mean((2, 3)), 0)
    surv_first = np.median(Sa[:, :, :, 0].mean(2), 0)
    ctx = {t: np.median(Ca[:, :, :, t].mean(2), 0) for t in CTX_T}
    gain = np.median(Ga.mean((2, 3)), 0)
    log.info("%d concepts. survival at t=0 by row %s: %s", K, rows, [round(float(x), 2) for x in surv_first])
    log.info("survival averaged over %d generated tokens: %s", N, [round(float(x), 2) for x in surv_mean])
    for t in CTX_T:
        log.info("context-only survival at t=%2d: %s", t, [round(float(x), 2) for x in ctx[t]])
    log.info("|D|/|delta| averaged over tokens: %s", [round(float(x), 2) for x in gain])
    eff = json.loads(args.steereff.read_text())["efficacy"]["concept pole"] if args.steereff.exists() else {}
    eff = {int(k): v for k, v in eff.items()}
    common = [L for L in rows if L in eff]
    if common:
        for name, arr in (
            ("survival t=0", surv_first),
            ("survival mean over t", surv_mean),
            ("context-only t=15", ctx[15]),
        ):
            a = np.array([arr[rows.index(L)] for L in common])
            e = np.array([eff[L] for L in common])
            r = np.corrcoef(np.log(np.clip(a, 1e-3, None)), np.log(e))[0, 1]
            log.info(
                "%-22s vs efficacy c_L over rows %s: log-log r %.2f; normalized at %d: %s vs c_L %s",
                name,
                common,
                r,
                ref,
                [round(float(x / arr[rows.index(ref)]), 2) for x in a],
                [round(float(x), 2) for x in e],
            )
    fig, axes = plt.subplots(1, 3, figsize=(26, 7))
    cmap = plt.get_cmap("viridis")
    ax = axes[0]
    for i, L in enumerate(rows):
        ax.plot(np.arange(N), surv_t[i], color=cmap(i / (R - 1)), lw=2, label=f"row {L}")
    ax.set_xlabel("generated position t")
    ax.set_ylabel("<h_steer - h_base, v(35)> / |delta| at the last row")
    ax.set_title("survival of the injected direction along the generation (median over concepts)")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(rows, surv_first, marker="o", lw=2, color="#9ca3af", label="survival at t = 0 (prompt position)")
    ax.plot(rows, surv_mean, marker="o", lw=2, color="#111827", label=f"survival averaged over {N} generated tokens")
    for t, col in zip((3, 15, 23), ("#93c5fd", "#2563eb", "#1e3a8a")):
        ax.plot(rows, ctx[t], marker="o", lw=1.5, ls="--", color=col, label=f"context-only survival at t = {t}")
    if common:
        ax.plot(
            common,
            [eff[L] * surv_mean[rows.index(ref)] for L in common],
            marker="s",
            lw=2,
            color="#dc2626",
            label=f"judged efficacy c_L (scaled to the row-{ref} mean survival)",
        )
    ax.axvspan(b0, b1, color="#f59e0b", alpha=0.12)
    ax.set_xlabel("injection row L")
    ax.set_title("row profile: current-token survival vs what the generation accumulates")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[2]
    for t, col in zip(CTX_T, ("#e5e7eb", "#93c5fd", "#60a5fa", "#2563eb", "#1e3a8a")):
        ax.plot(rows, ctx[t] / np.clip(surv_t[:, t], 1e-6, None), marker="o", color=col, label=f"t = {t}")
    ax.axvspan(b0, b1, color="#f59e0b", alpha=0.12)
    ax.set_xlabel("injection row L")
    ax.set_title("share of the survival at t that arrives through the context (earlier injected positions)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.figs / "S4-gensurv.png", dpi=170)
    log.info("wrote %s", args.figs)


if __name__ == "__main__":
    main()
