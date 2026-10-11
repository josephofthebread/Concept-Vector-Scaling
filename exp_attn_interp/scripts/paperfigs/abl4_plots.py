#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("ABL4_FIGDATA", HERE / "figdata"))
FIG = HERE.parent / "Figures"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 12,
        "axes.titlesize": 12.5,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10,
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
    }
)
BLUE, RED, GREY, GREEN, PURPLE, AMBER = "#2166ac", "#b2182b", "#7f7f7f", "#1b7837", "#762a83", "#f4a582"
INK = "#222222"


def panel_label(ax, s, dx=-0.09, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")


def band(ax, lo, hi, label=None, color=AMBER):
    ax.axvspan(lo - 0.5, hi + 0.5, color=color, alpha=0.30, lw=0, zorder=0)
    if label:
        ax.text(
            (lo + hi) / 2,
            0.98,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9.5,
            color="#8a4b1a",
        )


def load():
    return (
        json.load(open(DATA / "xdecomp_summary.json")),
        json.load(open(DATA / "reliab.json")),
        json.load(open(DATA / "topiclane.json")),
        json.load(open(DATA / "tokan.json")),
        np.load(DATA / "vargain.npz", allow_pickle=True),
    )


def E_law(m, a, clip=0.02):
    m, a = np.asarray(m), np.asarray(a)
    B = len(m)
    P = np.concatenate([[0.0], np.cumsum(m + a)[:-1]])
    g = m / np.clip(P, clip, None)
    mp = np.clip(m, 0, None)
    E = np.zeros(B)
    for r in range(B):
        seed, e = 1.0, 0.0
        for b in range(r, B):
            e += mp[b] * seed
            seed *= 1 + g[b]
        E[r] = e
    return E


def fig_crossing(xd, rl):
    """Plot b*, the law optimum, the probe optimum and the judged best layer across eight checkpoints."""
    order = [
        "Llama-3.1-8B-Instruct",
        "Qwen3-8B",
        "gemma-2-9b-it",
        "deepseek-llm-7b-chat",
        "Qwen3-4B",
        "Qwen2.5-7B",
        "Qwen2.5-3B-Instruct",
        "Qwen1.5-0.5B",
    ]
    disp = {
        "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
        "Qwen3-8B": "Qwen3-8B",
        "gemma-2-9b-it": "gemma-2-9b",
        "deepseek-llm-7b-chat": "deepseek-7b",
        "Qwen3-4B": "Qwen3-4B",
        "Qwen2.5-7B": "Qwen2.5-7B (base)",
        "Qwen2.5-3B-Instruct": "Qwen2.5-3B",
        "Qwen1.5-0.5B": "Qwen1.5-0.5B (base)",
    }
    judged = {"Qwen3-8B": "qwen", "Llama-3.1-8B-Instruct": "llama", "gemma-2-9b-it": "gemma"}
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ys = np.arange(len(order))[::-1]
    for y, mk in zip(ys, order):
        q = xd[mk]
        B = q["n_blocks"]
        bstar = q["crossing_last"]["first_positive"] / B
        probe = q["snr_argmax_row"] / B
        E = E_law(q["mlp_share_last_median"], q["attn_share_last_median"])
        eb = (int(np.argmax(E[1:B])) + 1) / B
        ax.plot([0, 1], [y, y], color="#e5e5e5", lw=6, solid_capstyle="round", zorder=0)
        ax.scatter(bstar, y, s=150, color=BLUE, zorder=3, label="sign crossing $b^{*}$" if y == ys[0] else None)
        ax.scatter(
            eb,
            y,
            s=95,
            color=INK,
            marker="D",
            zorder=4,
            label="efficacy-law optimum $\\arg\\max E$" if y == ys[0] else None,
        )
        ax.scatter(
            probe, y, s=150, color=GREEN, marker="^", zorder=3, label="probe SNR optimum" if y == ys[0] else None
        )
        if mk in judged:
            tag = judged[mk]
            rows = np.array(rl[tag]["rows"])
            lc = np.array(rl[tag]["concept_pole_a0.5"]["mean"])
            la = np.array(rl[tag]["antagonist_pole_a0.5"]["mean"])
            jbr = rows[int(np.argmax(lc + la))]
            print(f"  crossing {mk}: judged best (both poles) = {jbr}")
            ax.scatter(
                jbr / B,
                y,
                s=230,
                facecolors="none",
                edgecolors=RED,
                linewidths=2.0,
                zorder=5,
                label="judged best layer" if y == ys[0] else None,
            )
    ax.axvspan(0.45, 0.68, color=AMBER, alpha=0.22, lw=0, zorder=0)
    ax.text(0.565, len(order) - 0.35, "middle third of the stack", ha="center", fontsize=10, color="#8a4b1a")
    ax.set_yticks(ys)
    ax.set_yticklabels([disp[m] for m in order])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("relative depth  (layer $/$ number of blocks)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", fontsize=9.5, ncol=2, bbox_to_anchor=(0.0, -0.02))
    ax.set_title(
        "One forward pass locates the layer in every model\n"
        "the crossing $b^{*}$ and the efficacy-law optimum coincide; the judged best layer and probe optimum lie within a few blocks",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG / "crossing.pdf")
    plt.close(fig)


def main():
    xd, rl, tl, tk, z = load()
    q = xd["Qwen3-8B"]
    gain = np.asarray(q["gain_running_median"])
    final_write = np.asarray(q["mlp_share_last_median"])
    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "pdf.fonttype": 42,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    ):
        fig, (left, right) = plt.subplots(1, 2, figsize=(5.5, 2.15), gridspec_kw={"width_ratios": [2, 1]})
        blocks = np.arange(3, len(gain))
        left.bar(blocks, gain[blocks], width=0.78, color=np.where(gain[blocks] < 0, BLUE, RED), linewidth=0)
        left.axhline(0, color=INK, lw=0.7)
        left.set(xlim=(2.5, 35.5), ylim=(-0.115, 0.095), xlabel="block", ylabel=r"$\tilde m_b$")
        left.set_xticks([5, 10, 15, 20, 25, 30, 35])

        focus = np.arange(9, 28)
        right.plot(focus, final_write[focus], color=PURPLE, lw=1.4, marker="o", markersize=2.2)
        right.axhline(0, color=INK, lw=0.7)
        right.set(xlim=(8.5, 27.5), ylim=(-0.0025, 0.0325), xlabel="block", ylabel="$m_b$")
        right.set_xticks([10, 15, 20, 25])
        right.set_yticks([0, 0.01, 0.02, 0.03])

        for axis in (left, right):
            axis.grid(axis="y", linestyle=":", linewidth=0.45, alpha=0.45)
            axis.set_axisbelow(True)
            for depth in (12, 18, 24):
                axis.axvline(depth, color="#777777", linestyle="--", linewidth=0.65, alpha=0.65, zorder=0.5)
        fig.tight_layout(pad=0.3, w_pad=1.2)
        fig.savefig(FIG / "abl4-overview.pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    print("abl4-overview.pdf")
    models = [
        ("Qwen3-8B", "qwen", "\\texttt{Qwen3-8B}", 0.92),
        ("Llama-3.1-8B-Instruct", "llama", "\\texttt{Llama-3.1-8B}", 0.87),
        ("gemma-2-9b-it", "gemma", "\\texttt{gemma-2-9b}", 0.94),
    ]
    bands = {"qwen": (19, 22), "llama": (13, 17), "gemma": (21, 24)}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for ax, (mk, tag, disp, pear) in zip(axes, models):
        m = np.array(xd[mk]["mlp_share_last_median"])
        a = np.array(xd[mk]["attn_share_last_median"])
        B = len(m)
        E = E_law(m, a)
        rows = np.array(rl[tag]["rows"])
        P = np.concatenate([[0.0], np.cumsum(m + a)[:-1]])
        gg = m / np.clip(P, 0.02, None)
        mp = np.clip(m, 0, None)
        S = np.array([np.prod(1 + np.clip(gg[r:], None, 0)) for r in range(B)])
        A = np.array([mp[r:].sum() for r in range(B)])
        lc = np.array(rl[tag]["concept_pole_a0.5"]["mean"])
        la = np.array(rl[tag]["antagonist_pole_a0.5"]["mean"])
        score = (lc + la) / 2.0
        x = np.arange(1, B)
        ax.plot(x, S[1:B] / S[1:B].max(), color="#9aa0a6", lw=1.6, ls=(0, (5, 2)), label="survival term only (control)")
        ax.plot(
            x,
            A[1:B] / A[1:B].max(),
            color="#9aa0a6",
            lw=1.6,
            ls=(0, (1, 1.5)),
            label="amplification term only (control)",
        )
        ax.plot(x, E[1:B] / E[1:B].max(), color=INK, lw=2.6, label="Predicted profile $E(r)$")
        lo, hi = bands[tag]
        band(ax, lo, hi, "judged: at least 90% of maximum")
        eb = int(np.argmax(E[1:B])) + 1
        ax.axvline(eb, color=INK, ls=(0, (2, 2)), lw=1.2)
        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)
        ax2.grid(False)
        ax2.plot(rows, score, color=RED, marker="o", ms=4.5, lw=1.8, label="Concept-expression score (both poles)")
        ax2.tick_params(axis="y", colors=RED)
        ax2.set_ylim(0, 2)
        jb = rows[int(np.argmax(score))]
        rp = np.corrcoef(E[rows], score)[0, 1]
        print(f"  efficacy {mk}: E peak {eb}, judged best {jb}, Pearson {rp:+.2f}")
        disp_clean = disp.replace("\\texttt{", "").replace("}", "")
        ax.set_title(f"{disp_clean}\nPredicted: {eb}  ·  judged: {jb}  ·  Pearson $r={rp:.2f}$", fontsize=11.5)
        ax.set_xlabel("injection layer $r$")
        if ax is axes[0]:
            ax.set_ylabel("Predicted profile $E(r)$ (rescaled)")
        ax2.set_ylabel("Concept-expression score (0–2)", color=RED)
        ax.set_ylim(0, 1.08)
        ax.set_xlim(max(1, rows[0] - 6), rows[-1] + 6)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if ax is axes[0]:
            ax.legend(h1 + h2, l1 + l2, loc="lower center", fontsize=8.2)
    fig.suptitle("Extraction writes predict concept expression across injection layers", fontsize=12.5, y=1.05)
    fig.text(
        0.5,
        -0.03,
        "Judged score: AXBench-adapted concept expression, $0$–$2$; $\\alpha=0.5$; mean over both vector orientations",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIG / "abl4-efficacy.pdf", bbox_inches="tight")
    plt.close(fig)
    print("abl4-efficacy.pdf")
    blocks = [10, 18, 22, 30]
    styles = {
        "tokpc": ("stream principal axes (512)", GREY, "o", 9, 0.5),
        "concept": ("concept vectors ($\\mu_c-\\mu_a$)", BLUE, "o", 30, 1.0),
        "topic": ("shared-situation direction", PURPLE, "o", 30, 1.0),
        "random": ("random directions", RED, "s", 30, 1.0),
        "u_gap": ("marker-word directions", GREEN, "D", 30, 1.0),
    }
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.3), sharey=False)
    for ax, b in zip(axes, blocks):
        names = [str(n) for n in z[f"b{b}_names"]]
        grp = np.array([n.split(":")[0] for n in names])
        lv = np.log10(np.clip(z[f"b{b}_var"], 1e-12, None))
        g = z[f"b{b}_gain_0.1"]
        r = np.argsort(np.argsort(lv))
        rg = np.argsort(np.argsort(g))
        rho = np.corrcoef(r, rg)[0, 1]
        for key, (lab, col, mk, s, al) in styles.items():
            sel = grp == key
            if sel.sum():
                ax.scatter(
                    lv[sel],
                    g[sel],
                    s=s,
                    c=col,
                    marker=mk,
                    alpha=al,
                    edgecolors="none",
                    label=lab if b == 10 else None,
                    zorder=2 if key == "tokpc" else 3,
                )
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(f"block {b}    (Spearman $={rho:+.2f}$)", fontsize=11.5)
        ax.set_xlabel("log$_{10}$ variance of token projections")
        if ax is axes[0]:
            ax.set_ylabel("Finite-difference MLP gain")
    axes[0].legend(loc="lower right", fontsize=8.5, handletextpad=0.3, frameon=True, facecolor="white", framealpha=0.95)
    axes[1].annotate("lowest correlation\n(block 18)", xy=(1.3, -0.02), fontsize=9, color=GREY, ha="center")
    fig.text(
        0.5,
        -0.02,
        "Perturbation size: $0.1$ times the median MLP-input-state norm; variance measured over the same story tokens",
        ha="center",
        fontsize=9,
    )
    fig.suptitle("MLP gain rises with token-level variance along a direction", fontsize=12.5, y=1.05)
    fig.tight_layout()
    fig.savefig(FIG / "abl4-variance.pdf", bbox_inches="tight")
    plt.close(fig)
    print("abl4-variance.pdf")
    gc = tl["gain_current"]
    B = len(gc["mlp_lean"]["p50"])
    x = np.arange(B)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax = axes[0]
    ax.axhline(0, color=INK, lw=0.8)
    ax.plot(x, gc["mlp_lean"]["p50"], color=BLUE, lw=2.4, label="pole contrast $\\mu_c-\\mu_a$")
    ax.plot(x, gc["mlp_topic"]["p50"], color=PURPLE, lw=2.0, ls="--", label="shared situation")
    band(ax, 19, 22, "layers 19–22")
    ax.set_xlim(2, B - 0.5)
    ax.set_ylim(-0.13, 0.11)
    ax.set_title("(A) MLP writes to both directions change sign together", fontsize=11.5)
    ax.set_xlabel("transformer block")
    ax.set_ylabel("MLP write along each direction")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.annotate(
        "both directions have similar MLP writes",
        xy=(11, gc["mlp_lean"]["p50"][11]),
        xytext=(15.5, -0.10),
        fontsize=9.5,
        color=INK,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.0),
    )
    sh = tl["share_final22"]
    axb = axes[1]
    axb.plot(x, sh["attn_lean"]["p50"], color=BLUE, lw=2.2, label="pole contrast")
    axb.plot(x, sh["attn_topic"]["p50"], color=PURPLE, lw=2.2, ls="--", label="shared situation")
    band(axb, 19, 22)
    axb.axvline(19, color=INK, ls=":", lw=1.0)
    axb.annotate(
        "block 19: $0.18$ contrast\nversus $0.11$ shared situation",
        xy=(19, sh["attn_lean"]["p50"][19]),
        xytext=(13, 0.23),
        ha="center",
        fontsize=9.3,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.0),
    )
    axb.set_title("(B) Attention favors the pole contrast at block 19", fontsize=11.5)
    axb.set_xlabel("transformer block")
    axb.set_ylabel("Attention write along each direction")
    axb.legend(loc="upper left", fontsize=9.5)
    fig.suptitle("Middle MLPs contract both components of a feature pair", fontsize=12.5, y=1.05)
    fig.tight_layout(w_pad=4.5)
    fig.savefig(FIG / "abl4-lanes.pdf", bbox_inches="tight")
    plt.close(fig)
    print("abl4-lanes.pdf")
    w = tk["window"]["pol_u"]
    R = len(w["corr_at_0"])
    x = np.arange(R)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    ax = axes[0]
    ax.plot(x, w["corr_at_0"], color=INK, lw=2.0, label="current word")
    ax.plot(x, w["corr_sum_past"], color=BLUE, lw=2.0, label="words already read (past)")
    ax.plot(x, w["corr_sum_future"], color=RED, lw=2.0, label="words to come (future)")
    band(ax, 19, 22)
    ax.axvline(22, color=INK, ls=":", lw=1.0)
    ax.annotate(
        "block 22:\npast = future",
        xy=(22, w["corr_sum_future"][22]),
        xytext=(28, 0.05),
        fontsize=9.3,
        color=INK,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.0),
    )
    ax.set_title("(A) Current, preceding, and later words", fontsize=11.5)
    ax.set_xlabel("transformer block")
    ax.set_ylabel("within-story correlation\nwith word polarity")
    ax.legend(loc="upper right", fontsize=9)
    S = np.array(tk["classes_snr_median"])
    axb = axes[1]
    for ci, col, lab in [(0, PURPLE, "punctuation"), (1, GREEN, "function words"), (2, BLUE, "content words")]:
        axb.plot(x, S[ci], color=col, lw=2.0, label=lab)
    axb.axvline(7, color=INK, ls=":", lw=1.0)
    band(axb, 19, 22)
    axb.annotate(
        "from block 7:\npunctuation > function > content",
        xy=(7, 0.62),
        xytext=(9.5, 1.95),
        fontsize=9.3,
        color=INK,
        ha="left",
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.0),
    )
    axb.set_title("(B) Probe signal-to-noise by token type", fontsize=11.5)
    axb.set_xlabel("transformer block")
    axb.set_ylabel("probe signal-to-noise")
    axb.legend(loc="lower right", fontsize=9)
    A = np.array(tk["latency"]["auroc_raw_mean"])
    P = tk["prefix"]
    axc = axes[2]
    for pi, col, lab in [(4, "#9ecae1", "32 tokens"), (5, RED, "64 tokens"), (6, "#08519c", "128 tokens")]:
        if pi < len(P):
            axc.plot(x, A[pi], color=col, lw=2.0, label=f"first {P[pi]} tokens")
    axc.axhline(0.95, color=INK, ls=(0, (2, 2)), lw=1.0)
    band(axc, 19, 22)
    axc.text(1, 0.955, "AUROC 0.95", fontsize=9, color=INK)
    axc.annotate(
        "0.95 from 64 tokens\nonly in blocks 18–27",
        xy=(22, 0.963),
        xytext=(24, 0.80),
        fontsize=9.3,
        color=RED,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.0),
    )
    axc.set_title("(C) AUROC from the first generated tokens", fontsize=11.5)
    axc.set_xlabel("transformer block")
    axc.set_ylabel("story AUROC from a prefix")
    axc.set_ylim(0.6, 1.0)
    axc.legend(loc="lower right", fontsize=9)
    fig.suptitle("What token projections reflect at each layer", fontsize=12.5, y=1.05)
    fig.tight_layout()
    fig.savefig(FIG / "abl4-token.pdf", bbox_inches="tight")
    plt.close(fig)
    print("abl4-token.pdf")


if __name__ == "__main__":
    main()
