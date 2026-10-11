import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.environ["ABL4_FIGDATA"] = str(HERE / "figdata")
sys.path.insert(0, str(HERE.parent.parent / "report_ICLR2027" / "tools"))
import abl4_plots as P  # noqa: E402
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
BLUE, RED, GREY, GREEN, PURPLE, AMBER, INK = P.BLUE, P.RED, P.GREY, P.GREEN, P.PURPLE, P.AMBER, P.INK
CTRL = "#9a9a9a"
STYLE = {
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
    "axes.titlepad": 3,
}
W = 5.5


def ygrid(ax):
    ax.grid(axis="y", linestyle=":", linewidth=0.45, alpha=0.45)
    ax.set_axisbelow(True)


def band(ax, lo, hi):
    ax.axvspan(lo - 0.5, hi + 0.5, color=AMBER, alpha=0.30, lw=0, zorder=0)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    xd, rl, tl, tk, z = P.load()
    with mpl.rc_context(STYLE):
        models = [
            ("Qwen3-8B", "qwen", "Qwen3-8B"),
            ("Llama-3.1-8B-Instruct", "llama", "Llama-3.1-8B"),
            ("gemma-2-9b-it", "gemma", "gemma-2-9b"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(W, 1.85))
        for i, (ax, (mk, tag, disp)) in enumerate(zip(axes, models)):
            m = np.array(xd[mk]["mlp_share_last_median"])
            a = np.array(xd[mk]["attn_share_last_median"])
            B = len(m)
            E = P.E_law(m, a)
            rows = np.array(rl[tag]["rows"])
            Pc = np.concatenate([[0.0], np.cumsum(m + a)[:-1]])
            gg = m / np.clip(Pc, 0.02, None)
            mp = np.clip(m, 0, None)
            S = np.array([np.prod(1 + np.clip(gg[r:], None, 0)) for r in range(B)])
            A = np.array([mp[r:].sum() for r in range(B)])
            score = (
                np.array(rl[tag]["concept_pole_a0.5"]["mean"]) + np.array(rl[tag]["antagonist_pole_a0.5"]["mean"])
            ) / 2
            x = np.arange(1, B)
            near_peak = rows[score >= 0.9 * score.max()]
            band(ax, int(near_peak.min()), int(near_peak.max()))
            ax.plot(x, S[1:B] / S[1:B].max(), color=CTRL, lw=1.0, ls=(0, (4, 1.6)), label="survival term only")
            ax.plot(x, A[1:B] / A[1:B].max(), color=CTRL, lw=1.0, ls=(0, (1, 1.2)), label="amplification term only")
            ax.plot(x, E[1:B] / E[1:B].max(), color=INK, lw=1.5, label="predicted profile $E(r)$")
            eb = int(np.argmax(E[1:B])) + 1
            ax.axvline(eb, color=INK, ls=(0, (2, 2)), lw=0.7)
            ax2 = ax.twinx()
            ax2.spines["right"].set_visible(True)
            ax2.spines["right"].set_color(RED)
            ax2.plot(rows, score, color=RED, marker="o", ms=2.2, lw=1.2, label="judged concept expression")
            ax2.set_ylim(0, 2)
            ax.set_ylim(0, 1.08)
            ax.set_xlim(max(1, rows[0] - 6), rows[-1] + 6)
            ax.set_title(disp)
            ax.set_xlabel("injection layer $r$")
            ax2.tick_params(axis="y", colors=RED, width=0.6)
            ygrid(ax)
            if i == 0:
                ax.set_ylabel("$E(r)$, rescaled")
            else:
                ax.tick_params(axis="y", labelleft=False)
            if i == 2:
                ax2.set_ylabel("concept expression (0–2)", color=RED)
            else:
                ax2.tick_params(axis="y", labelright=False)
            if i == 0:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                hh, ll = h1[::-1] + h2, l1[::-1] + l2
        fig.tight_layout(pad=0.3, w_pad=0.9)
        fig.legend(
            hh,
            ll,
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, 0.0),
            handlelength=2.2,
            columnspacing=1.4,
            handletextpad=0.5,
            borderaxespad=0,
        )
        save(fig, "abl4-efficacy")
        gc = tl["gain_current"]
        B = len(gc["mlp_lean"]["p50"])
        x = np.arange(B)
        sh = tl["share_final22"]
        fig, (ax, axb) = plt.subplots(1, 2, figsize=(W, 2.0))
        band(ax, 19, 22)
        ax.axhline(0, color=INK, lw=0.6)
        ax.plot(x, gc["mlp_lean"]["p50"], color=BLUE, lw=1.3, label="pole contrast")
        ax.plot(x, gc["mlp_topic"]["p50"], color=PURPLE, lw=1.3, ls=(0, (3.5, 1.5)), label="shared situation")
        ax.set_xlim(2, B - 0.5)
        ax.set_ylim(-0.12, 0.12)
        ax.set_xticks([5, 10, 15, 20, 25, 30, 35])
        ax.set_xlabel("block")
        ax.set_ylabel("MLP write along each direction")
        ax.text(20.5, 0.117, "layers 19–22", ha="center", va="top", fontsize=6, color="#8a4b1a")
        ygrid(ax)
        band(axb, 19, 22)
        axb.plot(x, sh["attn_lean"]["p50"], color=BLUE, lw=1.3, label="pole contrast")
        axb.plot(x, sh["attn_topic"]["p50"], color=PURPLE, lw=1.3, ls=(0, (3.5, 1.5)), label="shared situation")
        axb.legend(loc="upper left", handlelength=2.0, labelspacing=0.3, borderaxespad=0.2)
        axb.annotate(
            "block 19: $0.18$ contrast\nvs $0.11$ shared situation",
            xy=(19, sh["attn_lean"]["p50"][19]),
            xytext=(0.8, 0.212),
            ha="left",
            va="center",
            fontsize=6,
            color=INK,
            arrowprops=dict(
                arrowstyle="-|>", color=INK, lw=0.6, mutation_scale=5, relpos=(0.8, 0.0), shrinkA=1, shrinkB=1.5
            ),
        )
        axb.set_xlim(-0.5, B - 0.5)
        axb.set_ylim(0, 0.31)
        axb.set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
        axb.set_xlabel("block")
        axb.set_ylabel("attention write along each direction")
        ygrid(axb)
        fig.tight_layout(pad=0.3, w_pad=1.6)
        save(fig, "abl4-lanes")
        w = tk["window"]["pol_u"]
        R = len(w["corr_at_0"])
        x = np.arange(R)
        fig, (ax, axb, axc) = plt.subplots(1, 3, figsize=(W, 1.95))
        leg = dict(handlelength=1.4, handletextpad=0.4, labelspacing=0.25, borderaxespad=0.2)
        band(ax, 19, 22)
        ax.plot(x, w["corr_sum_past"], color=BLUE, lw=1.2, label="past")
        ax.plot(x, w["corr_at_0"], color=INK, lw=1.2, label="current")
        ax.plot(x, w["corr_sum_future"], color=RED, lw=1.2, label="future")
        ax.set_ylim(-0.05, 0.5)
        ax.set_xlim(-1, R)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
            **{**leg, "handlelength": 0.9, "columnspacing": 0.7, "handletextpad": 0.3, "borderaxespad": 0.1},
        )
        ax.set_ylabel("correlation with word polarity")
        S = np.array(tk["classes_snr_median"])
        band(axb, 19, 22)
        for ci, col, lab in [(0, PURPLE, "punctuation"), (1, GREEN, "function"), (2, BLUE, "content")]:
            axb.plot(x, S[ci], color=col, lw=1.2, label=lab)
        axb.axvline(7, color=INK, ls=":", lw=0.6)
        axb.text(7.8, 2.3, "block 7", fontsize=6, color=INK, ha="left", va="center")
        axb.set_ylim(0, 2.45)
        axb.set_xlim(-1, R)
        axb.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
            **{**leg, "handlelength": 0.9, "columnspacing": 0.7, "handletextpad": 0.3, "borderaxespad": 0.1},
        )
        axb.set_ylabel("probe signal-to-noise")
        A = np.array(tk["latency"]["auroc_raw_mean"])
        Pf = tk["prefix"]
        band(axc, 19, 22)
        for pi, col in [(4, "#92c5de"), (5, RED), (6, BLUE)]:
            axc.plot(x, A[pi], color=col, lw=1.2, label=f"{Pf[pi]} tokens" if pi == 6 else f"{Pf[pi]}")
        axc.axhline(0.95, color=INK, ls=(0, (2, 2)), lw=0.6)
        axc.set_ylim(0.72, 1.0)
        axc.set_xlim(-1, R)
        axc.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
            **{**leg, "handlelength": 0.9, "columnspacing": 0.7, "handletextpad": 0.3, "borderaxespad": 0.1},
        )
        axc.set_ylabel("story AUROC from a prefix")
        for a_ in (ax, axb, axc):
            a_.set_xlabel("block")
            a_.set_xticks([0, 10, 20, 30])
            ygrid(a_)
        fig.tight_layout(pad=0.3, w_pad=1.4)
        save(fig, "abl4-token")
        blocks = [10, 18, 22, 30]
        styles = {
            "tokpc": ("principal axes (512)", "#b0b0b0", "o", 2.5, 0.6),
            "concept": ("concept vectors", BLUE, "o", 5, 0.9),
            "topic": ("shared situation", PURPLE, "o", 5, 0.9),
            "random": ("random", RED, "s", 5, 0.9),
            "u_gap": ("marker words", GREEN, "D", 4.5, 0.9),
        }
        fig, axes = plt.subplots(1, 4, figsize=(W, 1.75))
        for ax, b in zip(axes, blocks):
            names = [str(n) for n in z[f"b{b}_names"]]
            grp = np.array([n.split(":")[0] for n in names])
            lv = np.log10(np.clip(z[f"b{b}_var"], 1e-12, None))
            g = z[f"b{b}_gain_0.1"]
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
                        linewidths=0,
                        label=lab,
                        zorder=2 if key == "tokpc" else 3,
                        rasterized=True,
                    )
            ax.axhline(0, color=INK, lw=0.6)
            ax.set_title(f"block {b}")
            ax.set_xlabel("log$_{10}$ variance")
            ygrid(ax)
        axes[0].set_ylabel("MLP gain $G_b$")
        fig.tight_layout(pad=0.3, w_pad=0.6)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(
            h,
            l,
            loc="upper center",
            ncol=5,
            bbox_to_anchor=(0.5, 0.0),
            handletextpad=0.2,
            columnspacing=1.0,
            markerscale=1.6,
            borderaxespad=0,
        )
        save(fig, "abl4-variance")
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "preview.tex"],
        cwd=HERE,
        capture_output=True,
        check=True,
    )
    print("wrote", OUT, "and", HERE / "preview.pdf")


if __name__ == "__main__":
    main()
