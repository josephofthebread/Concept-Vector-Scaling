from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt

COSINE = {
    "warmth": None,
    "empathy": 0.945,
    "emotional availability": 0.887,
    "mentalizing": 0.884,
    "empathetic humor": 0.868,
    "comfort with intimacy": 0.858,
}
COLOR = {
    "warmth": "#d62728",
    "empathy": "#ff7f0e",
    "emotional availability": "#2ca02c",
    "mentalizing": "#1f77b4",
    "empathetic humor": "#9467bd",
    "comfort with intimacy": "#8c564b",
}
PANELS = [
    (Path("work/pca/base-raw"), "Meta-Llama-3.1-8B (base), raw"),
    (Path("work/pca/instruct-assistant"), "Meta-Llama-3.1-8B-Instruct, assistant"),
]
LAYER = 18


def load(src: Path) -> tuple[dict[str, list[tuple[float, float]]], dict]:
    points: dict[str, list[tuple[float, float]]] = {}
    with (src / f"points-L{LAYER}.csv").open() as handle:
        for row in csv.DictReader(handle):
            points.setdefault(row["concept"], []).append((float(row["pc1"]), float(row["pc2"])))
    return points, json.loads((src / f"summary-L{LAYER}.json").read_text())


figure, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), sharex=True, sharey=True)
for axis, (src, title) in zip(axes, PANELS, strict=True):
    points, summary = load(src)
    for concept in COSINE:
        xs = [p[0] for p in points[concept]]
        ys = [p[1] for p in points[concept]]
        axis.scatter(xs, ys, s=5, alpha=0.22, linewidths=0, color=COLOR[concept], rasterized=False)
    for concept in COSINE:
        xs = [p[0] for p in points[concept]]
        ys = [p[1] for p in points[concept]]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        axis.scatter([cx], [cy], s=190, color=COLOR[concept], edgecolor="white", linewidths=1.8, zorder=5)
    axis.set_title(
        f"{title}\nlayer {LAYER}   ·   {summary['stories']} stories   ·   PR = {summary['participation_ratio']:.0f}",
        fontsize=10,
    )
    axis.set_xlabel(f"PC1  ({summary['pc1_variance'] * 100:.1f}% of variance)")
    axis.axhline(0, color="#999999", linewidth=0.5, zorder=0)
    axis.axvline(0, color="#999999", linewidth=0.5, zorder=0)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)

_, left_summary = load(PANELS[0][0])
axes[0].set_ylabel(f"PC2  ({left_summary['pc2_variance'] * 100:.1f}% of variance)")

handles = [
    plt.Line2D(
        [],
        [],
        marker="o",
        linestyle="",
        markersize=8,
        color=COLOR[c],
        label=c if COSINE[c] is None else f"{c}   (cos {COSINE[c]:.3f})",
    )
    for c in COSINE
]
figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
figure.suptitle('Per-story concept vectors: "warmth vs coldness" and its five nearest siblings', fontsize=12)
figure.text(
    0.5,
    0.905,
    "each point is one story  ·  large circles are concept means  ·  "
    "the two axes together hold under 9% of the total spread",
    ha="center",
    fontsize=9,
    color="#555555",
)
figure.tight_layout(rect=(0, 0.06, 1, 0.90))
out = Path("work/pca/warmth6-pca-L18.svg")
figure.savefig(out, format="svg", bbox_inches="tight")
print(f"wrote {out}")
