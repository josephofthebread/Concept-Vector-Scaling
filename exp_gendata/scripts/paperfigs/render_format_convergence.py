import io
import json
import sys
import tarfile
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from safetensors import safe_open
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
REVIVE = ROOT / ".bak" / "CONVERGENCE_REVIVE"
MODE = sys.argv[1] if len(sys.argv) > 1 else "terciles"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "format-convergence.pdf"
NAMES = ["raw", "paired", "user", "assistant", "paired-\nassistant"]
PAIRS = list(combinations(range(5), 2))
CMAP = plt.get_cmap("Reds")
MODELS = [
    ("Qwen3-8B", "Qwen_Qwen3-8B--assistant--semantic", "Qwen3-8B", "#2166ac", 22),
    (
        "Meta-Llama-3.1-8B-Instruct",
        "unsloth_Meta-Llama-3.1-8B-Instruct--assistant--semantic",
        "Llama-3.1-8B",
        "#d87800",
        14,
    ),
    ("gemma-2-9b-it", "google_gemma-2-9b-it--assistant--semantic", "gemma-2-9b", "#2e8b57", 22),
]
plt.rcParams.update(
    {
        "font.size": 6.5,
        "pdf.fonttype": 42,
        "axes.titlesize": 7,
        "axes.labelsize": 6.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5.8,
    }
)


def curves(tag, layer):
    rows, grid = [], None
    with tarfile.open(REVIVE / "results" / "convcurve" / "raw" / f"{tag}.tar.gz") as tf:
        for m in tf.getmembers():
            if m.name.endswith(".npz"):
                z = np.load(io.BytesIO(tf.extractfile(m).read()))
                rows.append((int(Path(m.name).stem.split("-")[1]), z["auroc"][layer]))
                grid = [int(n) for n in z["grid"]]
    rows.sort()
    return np.stack([r[1] for r in rows]), grid


def c6(stem, layer):
    with safe_open(str(REVIVE / "inputs" / "convergence" / f"{stem}.safetensors"), framework="np") as h:
        return h.get_tensor("q_q")[layer].mean(0)


def main():
    z = np.load(REVIVE / "results" / "formats" / "format-cosines.npz")
    keys = ["Llama-3.1-8B_L21", "gemma-2-9b_L28"]
    means = np.concatenate([z[k].mean(1) for k in keys])
    norm = Normalize(vmin=means.min(), vmax=means.max())

    fig = plt.figure(figsize=(6.9, 2.45))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[1, 1, 0.12, 1.25],
        height_ratios=[1, 0.05],
        wspace=0.08,
        hspace=0.12,
        left=0.01,
        right=0.99,
        top=0.97,
        bottom=0.12,
    )
    heat_axes = []
    for i, key in enumerate(keys):
        ax = fig.add_subplot(gs[0, i])
        mean = np.full((5, 5), np.nan)
        for k, (a, b) in enumerate(PAIRS):
            mean[b, a] = z[key][k].mean()
        im = ax.imshow(mean, cmap=CMAP, norm=norm)
        for r in range(5):
            for c in range(r + 1):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec="0.35", lw=0.7, clip_on=False))
                ax.text(
                    c,
                    r,
                    NAMES[r] if r == c else f"{mean[r, c]:.3f}",
                    ha="center",
                    va="center",
                    rotation=45 if r == c else 0,
                    fontsize=5.0 if r == c else 5.6,
                )
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(4.5, -0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        heat_axes.append(ax)
    cax = fig.add_subplot(gs[1, :2])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("mean cosine between formats", labelpad=1)
    cb.ax.tick_params(length=1.5, pad=1)

    ax = fig.add_subplot(gs[:, 3])
    info = {}
    if MODE == "terciles":
        per = {"fast": [], "middle": [], "slow": []}
        for tag, stem, label, color, layer in MODELS:
            a, grid = curves(tag, layer)
            c = c6(stem, layer)
            lo, hi = np.percentile(c, [100 / 3, 200 / 3])
            for name, m in (("fast", c >= hi), ("middle", (c > lo) & (c < hi)), ("slow", c <= lo)):
                per[name].append(np.median(a[m], axis=0))
        cols = {"fast": "#1a9850", "middle": "#7f7f7f", "slow": "#d73027"}
        for name, arr in per.items():
            arr = np.stack(arr)
            ax.fill_between(grid, arr.min(0), arr.max(0), color=cols[name], alpha=0.18, lw=0)
            ax.plot(
                grid, np.median(arr, 0), color=cols[name], lw=1.3, marker="o", ms=2, label=f"{name}-converging third"
            )
            info[name] = dict(zip(grid, np.median(arr, 0).round(3).tolist()))
        ax.set_ylabel("median held-out AUROC", labelpad=1)
        ax.legend(loc="lower right", frameon=False)
    else:
        for tag, stem, label, color, layer in MODELS:
            a, grid = curves(tag, layer)
            c = c6(stem, layer)
            rho = [spearmanr(c, a[:, i])[0] for i in range(len(grid))]
            ax.plot(grid, rho, color=color, lw=1.3, marker="o", ms=2, label=label)
            info[label] = dict(zip(grid, np.round(rho, 3).tolist()))
        ax.axvline(8, color="k", lw=0.6, ls=":")
        ax.set_ylabel(r"Spearman($\bar c_6$, AUROC$_N$)", labelpad=1)
        ax.legend(loc="lower left", frameon=False)
    ax.set_xscale("log", base=2)
    ax.set_xticks(grid)
    ax.set_xticklabels([rf"$2^{{{int(np.log2(n))}}}$" for n in grid])
    ax.set_xlabel("stories per pole, $N$", labelpad=1)
    ax.grid(True, ls=":", lw=0.45, alpha=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.canvas.draw()
    left, right, bar = heat_axes[0].get_position(), heat_axes[1].get_position(), cax.get_position()
    cax.set_position([left.x0, bar.y0, right.x1 - left.x0, bar.height])
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
