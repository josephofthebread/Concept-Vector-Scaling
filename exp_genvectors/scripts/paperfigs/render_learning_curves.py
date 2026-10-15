import io
import json
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / ".bak" / "CONVERGENCE_REVIVE" / "results" / "convcurve" / "raw"
ONTOLOGY = ROOT / ".bak" / "ood-study" / "semantic.ontology.json"
OUT = Path(__file__).resolve().parent / "learning-curves.pdf"
MODELS = [("Qwen3-8B", 22), ("Meta-Llama-3.1-8B-Instruct", 14), ("gemma-2-9b-it", 22)]
CLASSES = [
    ("Uniformity vs Diversity", "uniformity vs diversity (fast)", "#1f77b4"),
    ("Leadership Archetypes", "leadership archetypes (fast)", "#2ca02c"),
    ("Honor, Shame & Facework", "honor, shame & facework (slow)", "#c0392b"),
    ("Classical Moral Dilemmas", "classical moral dilemmas (slow)", "#e08214"),
]
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "pdf.fonttype": 42,
    }
)


def main():
    classes = {int(p["pair_id"]): p["class_name"] for p in json.loads(ONTOLOGY.read_text())["pairs"]}
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (tag, layer) in zip(axes, MODELS):
        rows, grid = [], None
        with tarfile.open(RAW / f"{tag}.tar.gz") as tf:
            for m in tf.getmembers():
                if m.name.endswith(".npz"):
                    z = np.load(io.BytesIO(tf.extractfile(m).read()))
                    rows.append((int(Path(m.name).stem.split("-")[1]), z["auroc"][layer]))
                    grid = [int(n) for n in z["grid"]]
        rows.sort()
        pids, a, g = np.array([r[0] for r in rows]), np.stack([r[1] for r in rows]), np.array(grid)
        keep = rng.choice(len(a), 220, replace=False)
        for i, n in enumerate(g):
            ax.scatter(
                n * np.exp(rng.uniform(-0.05, 0.05, len(keep))),
                a[keep, i],
                s=0.6,
                color="#b9bdc5",
                alpha=0.35,
                linewidths=0,
                rasterized=True,
                zorder=1,
            )
        ax.fill_between(g, np.percentile(a, 5, 0), np.percentile(a, 95, 0), color="grey", alpha=0.15, lw=0, zorder=2)
        ax.plot(g, np.median(a, 0), color="k", ls="--", lw=1.1, zorder=4)
        cls = np.array([classes[int(p)] for p in pids])
        for prefix, _, color in CLASSES:
            m = np.array([s.startswith(prefix) for s in cls])
            ax.plot(g, a[m].mean(0), color=color, lw=1.5, marker="o", ms=2.2, zorder=5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(g)
        ax.set_xticklabels([rf"$2^{{{int(np.log2(n))}}}$" for n in g])
        ax.set_xlabel("stories per pole, $N$")
        ax.set_ylim(0.45, 1.01)
        ax.grid(True, ls=":", lw=0.45, alpha=0.45)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("held-out AUROC")
    handles = [Line2D([], [], color=c, lw=1.5, marker="o", ms=2.2, label=lab) for _, lab, c in CLASSES]
    handles += [
        Line2D([], [], color="k", ls="--", lw=1.1, label="median"),
        Line2D([], [], color="grey", lw=5, alpha=0.3, label="5th--95th percentiles"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
