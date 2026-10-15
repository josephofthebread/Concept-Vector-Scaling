from __future__ import annotations

from pathlib import Path

import numpy as np

BOOTSTRAP_SAMPLES = 2_000


def pair_table(vector_dir: Path):
    """Read either historical ``pair_id`` or published ``pair`` metadata."""
    import pandas as pd

    table = pd.read_parquet(vector_dir / "pairs.parquet")
    identifier = "pair_id" if "pair_id" in table.columns else "pair"
    if identifier not in table.columns:
        raise ValueError(f"{vector_dir / 'pairs.parquet'} has neither pair_id nor pair")
    return table.set_index(identifier, drop=False)


def outcome_statistics(
    pair_ids: np.ndarray,
    raw: np.ndarray,
    correct: np.ndarray,
    trace_benchmarks: np.ndarray,
    wanted: set[int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Standardize full-trace means and compare outcomes, balancing benchmarks."""
    if wanted is not None:
        keep = [index for index, pair in enumerate(pair_ids) if int(pair) in wanted]
        if len(keep) < 2:
            raise SystemExit(f"Only {len(keep)} of the chosen concepts are present")
        pair_ids, raw = pair_ids[keep], raw[:, keep]

    benchmark_masks = {name: trace_benchmarks == name for name in sorted(set(trace_benchmarks))}
    usable = {
        name: mask
        for name, mask in benchmark_masks.items()
        if np.sum(mask & correct) >= 2 and np.sum(mask & ~correct) >= 2
    }
    if not usable:
        raise SystemExit("No benchmark has at least two correct and two incorrect traces")

    z = np.empty_like(raw)
    for mask in benchmark_masks.values():
        std = raw[mask].std(axis=0)
        std[std == 0] = 1.0
        z[mask] = (raw[mask] - raw[mask].mean(axis=0)) / std
    right_mean = np.mean([z[mask & correct].mean(axis=0) for mask in usable.values()], axis=0)
    wrong_mean = np.mean([z[mask & ~correct].mean(axis=0) for mask in usable.values()], axis=0)
    return pair_ids, z, right_mean, wrong_mean, right_mean - wrong_mean, usable


def bootstrap_band(stacks: list[np.ndarray], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap within benchmark, then give each benchmark equal weight."""
    draws = np.empty((BOOTSTRAP_SAMPLES, stacks[0].shape[1]), dtype=np.float32)
    batch = 100
    for start in range(0, BOOTSTRAP_SAMPLES, batch):
        stop = min(start + batch, BOOTSTRAP_SAMPLES)
        batch_draws = []
        for stack in stacks:
            indices = rng.integers(0, len(stack), size=(stop - start, len(stack)))
            batch_draws.append(stack[indices].mean(axis=1))
        draws[start:stop] = np.mean(batch_draws, axis=0)
    return tuple(np.quantile(draws, [0.025, 0.975], axis=0))
