from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path

import numpy as np
from safetensors import safe_open

from exp_gendata.artifacts import atomic_safetensors

log = logging.getLogger("convergence")

SINGLES = tuple((b,) for b in range(4))
HALVES = tuple(combinations(range(4), 2))
FULL = (0, 1, 2, 3)


def families() -> dict[str, list[tuple[tuple[int, ...], tuple[int, ...]]]]:
    return {
        "q_q": [(a, b) for a, b in combinations(SINGLES, 2)],
        "h_h": [(a, tuple(b for b in FULL if b not in a)) for a in HALVES if 0 in a],
        "q_h": [(a, h) for a in SINGLES for h in HALVES],
        "q_f": [(a, FULL) for a in SINGLES],
        "h_f": [(h, FULL) for h in HALVES],
    }


SUBSETS = tuple(sorted({*SINGLES, *HALVES, FULL}))
RAW_DEFINITIONS = ("diff", "concept_centered", "antagonist_centered")


def subset_vectors(sums: np.ndarray, counts: np.ndarray, subset: tuple[int, ...]) -> dict[str, np.ndarray]:
    """The three raw definitions [pair, hidden] estimated from the given buckets only."""
    chosen = list(subset)
    mu = sums[:, :, chosen].sum(axis=2) / counts[:, :, chosen].sum(axis=2)[..., None]
    corpus = sums[:, :, chosen].sum(axis=(0, 1, 2)) / counts[:, :, chosen].sum()
    return {
        "diff": mu[:, 0] - mu[:, 1],
        "concept_centered": mu[:, 0] - corpus,
        "antagonist_centered": mu[:, 1] - corpus,
    }


def subset_diffs(sums: np.ndarray, counts: np.ndarray) -> dict[tuple[int, ...], np.ndarray]:
    """diff vectors [pair, hidden] for every bucket subset used by any family."""
    return {subset: subset_vectors(sums, counts, subset)["diff"] for subset in SUBSETS}


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30)


TITLES = {
    "q_q": "0.25 vs 0.25 (disjoint)",
    "h_h": "0.5 vs 0.5 (disjoint)",
    "q_h": "0.25 vs 0.5",
    "q_f": "0.25 vs 1.0",
    "h_f": "0.5 vs 1.0",
}
BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#1a1a2e"


def figure(tag: str, scores: dict[str, np.ndarray]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = scores["q_q"].shape[0]
    depth = np.arange(layers) / (layers - 1)
    q_h_overlap = np.array([len(set(a) & set(h)) > 0 for a in SINGLES for h in HALVES])
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), sharey=True)
    for axis, name in zip(axes, TITLES):
        data = scores[name]
        groups = [(data, BLUE, "")]
        if name == "q_h":
            groups = [
                (data[:, ~q_h_overlap], BLUE, "disjoint"),
                (data[:, q_h_overlap], ORANGE, "overlapping"),
            ]
        for values, color, label in groups:
            flat = values.reshape(layers, -1)
            x = np.repeat(depth, flat.shape[1])
            axis.scatter(x, flat.ravel(), s=1.5, alpha=0.03, color=color, rasterized=True, linewidths=0)
            axis.plot(depth, np.median(flat, axis=1), color=color, linewidth=2, label=label or None)
        axis.set_title(TITLES[name], fontsize=11, color=INK)
        axis.set_xlabel("fractional depth")
        axis.grid(color="#e6e6ef", linewidth=0.5)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(False)
        if name == "q_h":
            axis.legend(loc="lower right", frameon=False, fontsize=9)
    axes[0].set_ylabel("cosine between subset diff vectors")
    fig.suptitle(f"{tag}: diff-vector agreement across bucket subsets (points = pairs, line = median)", fontsize=12)
    fig.tight_layout()
    return fig


ALL_DEFINITIONS = (
    "diff",
    "concept_centered",
    "antagonist_centered",
    "whitened_diff",
    "whitened_concept_centered",
    "whitened_antagonist_centered",
)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--captures", type=Path, default=Path("work/captures"))
    parser.add_argument("--out", type=Path, default=Path("work/convergence"))
    parser.add_argument("--tag", help="process a single run directory instead of every semantic capture")
    parser.add_argument("--definitions", nargs="+", default=list(ALL_DEFINITIONS), choices=ALL_DEFINITIONS)
    parser.add_argument("--shrinkage", type=float, default=0.05)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    captures = (
        [args.captures / args.tag / "shard-000.safetensors"]
        if args.tag
        else sorted(args.captures.glob("*--semantic/shard-000.safetensors"))
    )
    for capture in captures:
        definitions = tuple(args.definitions)
        from scipy.linalg import cho_factor, cho_solve

        whitened_wanted = tuple(d for d in definitions if d.startswith("whitened_"))
        raw_wanted = tuple(d for d in definitions if not d.startswith("whitened_"))
        if whitened_wanted:
            with safe_open(str(capture), framework="np") as probe:
                if probe.get_slice("sums").get_dtype() in ("F16", "float16"):
                    raise SystemExit(
                        f"{capture}: whitened definitions require float32 sums; "
                        "the packed capture stores fp16 means. Refusing to produce invalid output."
                    )
        with safe_open(str(capture), framework="np") as handle:
            manifest = json.loads(handle.metadata()["manifest"])
            counts = handle.get_tensor("counts").astype(np.float32)
            stored = handle.get_tensor("sums")
            sums = (
                stored.astype(np.float32) * np.maximum(counts, 1)[None, ..., None]
                if stored.dtype == np.float16
                else stored.astype(np.float32)
            )
            layers, pairs = sums.shape[0], sums.shape[1]
            moments_slice = handle.get_slice("moments_tri") if whitened_wanted else None
            if moments_slice is not None:
                hidden = sums.shape[-1]
                tri_rows, tri_cols = np.triu_indices(hidden)

            spec = families()
            scores = {
                definition: {
                    name: np.empty((layers, len(combos), pairs), dtype=np.float32) for name, combos in spec.items()
                }
                for definition in definitions
            }
            for layer in range(layers):
                vectors: dict[str, dict[tuple[int, ...], np.ndarray]] = {d: {} for d in definitions}
                base = {subset: subset_vectors(sums[layer], counts, subset) for subset in SUBSETS}
                for subset in SUBSETS:
                    for definition in raw_wanted:
                        vectors[definition][subset] = base[subset][definition]
                if whitened_wanted:
                    triangle = np.asarray(moments_slice[layer]).astype(np.float64)
                    moments = np.zeros((triangle.shape[0], hidden, hidden))
                    moments[:, tri_rows, tri_cols] = triangle
                    moments[:, tri_cols, tri_rows] = triangle
                    for subset in SUBSETS:
                        chosen = list(subset)
                        group_counts = counts[:, :, chosen].sum(axis=2).reshape(-1)
                        group_sums = sums[layer][:, :, chosen].sum(axis=2).reshape(-1, hidden).astype(np.float64)
                        live = group_counts > 0
                        means = group_sums[live] / group_counts[live, None]
                        centered = moments[chosen].sum(axis=0) - means.T @ group_sums[live]
                        covariance = centered / group_counts[live].sum()
                        covariance = (covariance + covariance.T) * 0.5
                        scale = np.trace(covariance) / hidden
                        stacked = np.concatenate(
                            [base[subset][d.removeprefix("whitened_")] for d in whitened_wanted], axis=0
                        ).astype(np.float64)
                        try:
                            factor = cho_factor(covariance + args.shrinkage * scale * np.eye(hidden), lower=True)
                            solved = cho_solve(factor, stacked.T).T
                        except np.linalg.LinAlgError:
                            eigenvalues, basis = np.linalg.eigh(covariance)
                            inverse = 1.0 / (np.maximum(eigenvalues, 0.0) + args.shrinkage * scale)
                            solved = (stacked @ basis) * inverse @ basis.T
                        for position, definition in enumerate(whitened_wanted):
                            vectors[definition][subset] = solved[position * pairs : (position + 1) * pairs]
                for definition in definitions:
                    for name, combos in spec.items():
                        for position, (left, right) in enumerate(combos):
                            scores[definition][name][layer, position] = cosine_rows(
                                vectors[definition][left], vectors[definition][right]
                            )
                if layer % 8 == 0:
                    log.info("%s layer %s/%s", capture.parent.name, layer, layers)

        for definition in definitions:
            metadata = {
                "manifest": json.dumps(
                    {
                        "model": manifest["model"],
                        "presentation": manifest["presentation"],
                        "dataset": manifest["dataset"],
                        "vector": definition,
                        "shrinkage": args.shrinkage if definition.startswith("whitened_") else None,
                        "axes": ["layer", "combo", "pair"],
                        "combos": {name: [[list(a), list(b)] for a, b in combos] for name, combos in spec.items()},
                    },
                    sort_keys=True,
                )
            }
            out = args.out / definition / f"{capture.parent.name}.safetensors"
            atomic_safetensors(scores[definition], out, metadata)
        log.info("%s done", capture.parent.name)


if __name__ == "__main__":
    main()
