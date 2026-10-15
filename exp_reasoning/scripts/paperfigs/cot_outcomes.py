from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from exp_reasoning.stats import bootstrap_band, outcome_statistics, pair_table  # noqa: E402

UNCERTAINTY_PAIRS = (985, 1009)
COT_PAIRS = (367, 960, 357, 909, 963, 598, 908, 253, 533, 1013, 657, 902, 703, 146, 878, 459)
TOP = 8
SEED = 20260916
CORRECT_COLOR = "#2673b8"
INCORRECT_COLOR = "#bd1f2d"
CONNECTOR_COLOR = "#cccccc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True, help="NPZ written by exp_reasoning.readout")
    parser.add_argument("--vector-dir", type=Path, required=True, help="Directory containing pairs.parquet")
    parser.add_argument("--out", type=Path, required=True, help="Output PDF or PNG")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.scores, allow_pickle=False) as data:
        pair_ids = data["pair_ids"].astype(np.int32)
        correct = data["correct"].astype(bool)
        trace_benchmarks = data["trace_benchmarks"].astype(str)
        temporal = data["binned_z"].astype(np.float32)
        full_trace = data["mean_cosine"].astype(np.float32)
        layer = int(data["layer"])

    benchmark_names = sorted(set(trace_benchmarks))
    expected = {"gpqa_diamond", "math_500"}
    if set(benchmark_names) != expected:
        raise SystemExit(f"Expected GPQA Diamond and MATH-500, found {benchmark_names}")
    if correct.sum() < 2 or (~correct).sum() < 2:
        raise SystemExit("Both outcome groups need at least two traces")

    position = {int(pair): index for index, pair in enumerate(pair_ids)}
    missing = sorted(set(UNCERTAINTY_PAIRS) - set(position))
    if missing:
        raise SystemExit(f"Uncertainty pairs absent from {args.scores}: {missing}")
    pairs = pair_table(args.vector_dir)

    ranked_ids, _, correct_mean, incorrect_mean, gap, _ = outcome_statistics(
        pair_ids, full_trace, correct, trace_benchmarks, set(COT_PAIRS)
    )
    order = np.argsort(-np.abs(gap))[:TOP]
    order = order[np.argsort(-gap[order])]

    rng = np.random.default_rng(SEED)
    bins = temporal.shape[2]
    x = (np.arange(bins) + 0.5) * 100 / bins
    profiles: dict[tuple[int, bool], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for pair in UNCERTAINTY_PAIRS:
        for outcome in (True, False):
            stacks = [
                temporal[(trace_benchmarks == benchmark) & (correct == outcome), position[pair]]
                for benchmark in benchmark_names
            ]
            if any(len(stack) < 2 for stack in stacks):
                raise SystemExit(f"Pair {pair} lacks traces for an outcome/benchmark group")
            mean = np.mean([stack.mean(axis=0) for stack in stacks], axis=0)
            low, high = bootstrap_band(stacks, rng)
            profiles[pair, outcome] = mean, low, high

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 9,
        }
    )
    figure = plt.figure(figsize=(10.8, 4.8))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.0, 1.05),
        left=0.07,
        right=0.985,
        bottom=0.18,
        top=0.82,
        wspace=0.80,
        hspace=0.52,
    )
    temporal_axes = [figure.add_subplot(grid[row, 0]) for row in range(2)]
    outcome_axis = figure.add_subplot(grid[:, 1])

    for panel, (axis, pair) in enumerate(zip(temporal_axes, UNCERTAINTY_PAIRS, strict=True)):
        for outcome, color in ((True, CORRECT_COLOR), (False, INCORRECT_COLOR)):
            mean, low, high = profiles[pair, outcome]
            axis.plot(x, mean, color=color, linewidth=1.5)
            axis.fill_between(x, low, high, color=color, alpha=0.16, linewidth=0)
        axis.axhline(0, color="black", linewidth=0.7, linestyle=":")
        axis.set_xlim(0, 100)
        axis.set_ylabel("signed z-mean")
        axis.set_title(f"({chr(ord('A') + panel)}) {pairs.loc[pair, 'concept']}", loc="left")
        axis.grid(color="#c9c9c9", linestyle=":", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    temporal_axes[0].tick_params(labelbottom=False)
    temporal_axes[1].set_xlabel("relative position within reasoning, %")

    labels = [
        f"{pairs.loc[int(ranked_ids[index]), 'concept']}\n"
        rf"$\it{{vs}}$ {pairs.loc[int(ranked_ids[index]), 'antagonist']}"
        for index in order
    ]
    y = np.arange(TOP)
    outcome_axis.hlines(y, incorrect_mean[order], correct_mean[order], color=CONNECTOR_COLOR, linewidth=2.1)
    outcome_axis.scatter(correct_mean[order], y, color=CORRECT_COLOR, s=19, zorder=3)
    outcome_axis.scatter(incorrect_mean[order], y, color=INCORRECT_COLOR, s=19, zorder=3)
    outcome_axis.axvline(0, color="black", linewidth=0.8, linestyle=":")
    outcome_axis.set_yticks(y, labels)
    outcome_axis.set_ylim(TOP - 0.5, -0.5)
    outcome_axis.set_xlim(-0.6, 0.68)
    outcome_axis.set_xticks(np.arange(-0.6, 0.61, 0.2))
    outcome_axis.set_xlabel("signed z-mean")
    outcome_axis.set_title("(C) concept activation by outcome", loc="left")
    outcome_axis.grid(axis="x", color="#c9c9c9", linestyle=":", linewidth=0.6)
    outcome_axis.spines[["top", "right", "left"]].set_visible(False)
    outcome_axis.tick_params(axis="y", length=0)

    figure.suptitle(
        f"Qwen3-8B · layer {layer} · thinking tokens · GPQA Diamond + MATH-500",
        y=0.965,
        fontsize=12,
    )
    correct_count = int(correct.sum())
    incorrect_count = int((~correct).sum())
    handles = [
        plt.Line2D([0], [0], color=CORRECT_COLOR, marker="o", markersize=3, label=f"correct (n={correct_count})"),
        plt.Line2D([0], [0], color=INCORRECT_COLOR, marker="o", markersize=3, label=f"incorrect (n={incorrect_count})"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.035))
    figure.savefig(args.out)
    plt.close(figure)

    print(f"{len(correct)} traces: {correct_count} correct / {incorrect_count} incorrect")
    print("Top CoT features by absolute benchmark-balanced outcome gap:")
    for index in order:
        pair = int(ranked_ids[index])
        print(f"  {pair:4d}  {gap[index]:+.3f}  {pairs.loc[pair, 'concept']} vs {pairs.loc[pair, 'antagonist']}")
    print(f"Figure -> {args.out}")


if __name__ == "__main__":
    main()
