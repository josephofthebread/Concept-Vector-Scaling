from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np

Array = np.ndarray


def means(sums: Array, counts: Array, buckets: Sequence[int] | None = None) -> Array:
    """Per-pole means. sums [layer,pair,side,bucket,hidden], counts [pair,side,bucket] -> [layer,pair,side,hidden]."""
    chosen = list(buckets) if buckets is not None else list(range(counts.shape[-1]))
    pooled = counts[:, :, chosen].sum(axis=-1)
    if np.any(pooled == 0):
        raise ValueError("an empty (pair, side) cell over the selected buckets")
    return sums[:, :, :, chosen].sum(axis=3) / pooled[None, :, :, None]


def contrasts(sums: Array, counts: Array, corpus_sum: Array) -> dict[str, Array]:
    """The three raw vector definitions, each [layer, pair, hidden]."""
    mu = means(sums, counts)
    stories = int(counts.sum())
    if stories == 0:
        raise ValueError("empty capture")
    corpus = corpus_sum / stories
    return {
        "diff": mu[:, :, 0] - mu[:, :, 1],
        "concept_centered": mu[:, :, 0] - corpus[:, None],
        "antagonist_centered": mu[:, :, 1] - corpus[:, None],
    }


def within_covariance(moment: Array, group_sums: Array, group_counts: Array) -> Array:
    """Pooled within-(pair, pole) covariance from an uncentered second moment [hidden, hidden]."""
    live = group_counts > 0
    if not live.any():
        raise ValueError("covariance has no observations")
    group_means = group_sums[live] / group_counts[live, None]
    centered = moment - group_means.T @ group_sums[live]
    covariance = centered / int(group_counts[live].sum())
    return (covariance + covariance.T) * 0.5


def whiten(vectors: Array, covariance: Array, shrinkage: float) -> Array:
    """(covariance + shrinkage * mean_eigenvalue * I)^-1 @ vectors, rows are vectors."""
    if shrinkage <= 0:
        raise ValueError("shrinkage must be positive for a rank-deficient covariance")
    scale = float(np.trace(covariance)) / covariance.shape[0]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"covariance has invalid mean eigenvalue {scale}")
    regularized = covariance + shrinkage * scale * np.eye(covariance.shape[0])
    return np.linalg.solve(regularized, vectors.T).T


def cosine(left: Array, right: Array) -> Array:
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.einsum("...i,...i->...", left, right) / np.maximum(denominator, 1e-30)


def disjoint_splits(buckets: int, size: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Unordered pairs of disjoint same-size bucket subsets."""
    if size <= 0 or 2 * size > buckets:
        raise ValueError("split sides must be non-empty and disjoint")
    splits = []
    for left in combinations(range(buckets), size):
        for right in combinations([b for b in range(buckets) if b not in left], size):
            if right > left:
                splits.append((left, right))
    return splits


def half_splits(buckets: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    return disjoint_splits(buckets, buckets // 2)


def bucket_covariance(moments: Array, sums: Array, counts: Array, layer: int, buckets: Sequence[int]) -> Array:
    """Σ_W estimated from the selected buckets only, at one layer."""
    chosen = list(buckets)
    group_sums = sums[layer][:, :, chosen].sum(axis=2).reshape(-1, sums.shape[-1])
    group_counts = counts[:, :, chosen].sum(axis=2).reshape(-1)
    return within_covariance(moments[layer, chosen].sum(axis=0), group_sums, group_counts)


def split_reliability(
    sums: Array, counts: Array, moments: Array, *, layer: int, shrinkage: float, size: int | None = None
) -> dict[str, Array]:
    """Raw and independently-whitened split cosines, each [split, pair].
    Each side is whitened by its own buckets' covariance; a shared one would inflate the cosine."""
    raw_scores, white_scores = [], []
    for left, right in disjoint_splits(counts.shape[-1], size or counts.shape[-1] // 2):
        left_diff = np.squeeze(np.diff(means(sums, counts, left)[layer], axis=1) * -1, axis=1)
        right_diff = np.squeeze(np.diff(means(sums, counts, right)[layer], axis=1) * -1, axis=1)
        raw_scores.append(cosine(left_diff, right_diff))
        left_white = whiten(left_diff, bucket_covariance(moments, sums, counts, layer, left), shrinkage)
        right_white = whiten(right_diff, bucket_covariance(moments, sums, counts, layer, right), shrinkage)
        white_scores.append(cosine(left_white, right_white))
    return {"raw": np.stack(raw_scores), "whitened": np.stack(white_scores)}
