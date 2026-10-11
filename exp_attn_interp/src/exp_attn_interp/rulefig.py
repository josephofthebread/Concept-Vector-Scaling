from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    C = np.load("work/formation/curves.npz")
    d = json.loads(Path("work/decompan.json").read_text())
    Lz = np.load("work/lexcontent/lens.npz")
    B = C["along_m"].shape[1]
    rm = np.median(C["along_m"] / np.sqrt(C["along_m"] ** 2 + C["perp_m"] ** 2 + 1e-12), 0)
    ra = np.median(C["along_a"] / np.sqrt(C["along_a"] ** 2 + C["perp_a"] ** 2 + 1e-12), 0)
    tg = np.median(C["to_gen"], 0)
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    x = np.arange(B)
    axes[0].plot(x, ra, "-o", ms=3, color="#2563eb", lw=2, label="attention write: cos with the running direction v̂(b)")
    axes[0].plot(x, rm, "-s", ms=3, color="#6b7280", lw=2, label="MLP write: cos with v̂(b)")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("cos(write, v̂(b))")
    axes[0].set_title("Qwen3-8B, 1,036 concept vectors decomposed exactly into per-block writes (median over concepts)")
    m = np.array(d["per_block_mlp_share_median"])
    a = np.array(d["per_block_attention_share_median"])
    axes[1].bar(x, m, color="#6b7280", label="MLP write along the final direction v̂(35) (share)")
    axes[1].plot(x, a, "-o", ms=3, color="#2563eb", label="attention write along v̂(35) (share)")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_ylabel("share of v(35)")
    axes[2].plot(
        np.arange(B + 1),
        tg,
        "-o",
        ms=3,
        color="#059669",
        lw=2,
        label="cos(16-story v̂(L), 2,000-story v̂(L)): reproducibility of the direction",
    )
    axes[2].plot(np.arange(B + 1), np.median(C["to_last"], 0), "--", color="#7c3aed", lw=1.5, label="cos(v̂(L), v̂(35))")
    axes[2].set_ylabel("cosine")
    axes[3].plot(
        x,
        np.median(Lz["gap_m"], 0),
        "-s",
        ms=3,
        color="#6b7280",
        lw=2,
        label="MLP write: logit-lens marker gap (concept − antagonist words, z)",
    )
    axes[3].plot(x, np.median(Lz["gap_a"], 0), "-o", ms=3, color="#2563eb", lw=2, label="attention write: marker gap")
    axes[3].plot(
        np.arange(B + 1),
        np.median(Lz["han_v"], 0) * 5,
        ":",
        color="#dc2626",
        lw=1.5,
        label="Mandarin share of v(L)'s promoted tokens (×5)",
    )
    axes[3].axhline(0, color="k", lw=0.6)
    axes[3].set_ylabel("z / share")
    axes[3].set_xlabel("block b (writes into row b+1)")
    for ax in axes:
        ax.axvspan(
            19,
            22,
            color="#f59e0b",
            alpha=0.2,
            label="judged steering band (rows 19–22, best 22)" if ax is axes[0] else None,
        )
        for p, lab in (
            (22, "learning-curve plateau"),
            (20, "separability peak"),
            (24, "dose-response / agent readout (2L/3)"),
        ):
            ax.axvline(p, color="#7c3aed", ls="--", lw=1, label=lab if ax is axes[0] else None)
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    Path("figures/formation").mkdir(parents=True, exist_ok=True)
    fig.savefig("figures/formation/A9-rule.png", dpi=170)


if __name__ == "__main__":
    main()
