from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

from exp_genvectors.presentation import Wrapping

MIN_POOLED_TOKENS = 16


@dataclass(frozen=True)
class Row:
    pair: int
    bucket: int
    genre: int
    language: int
    generator: int
    concept_ids: tuple[int, ...]
    antagonist_ids: tuple[int, ...]
    concept_hit: int = 0
    antagonist_hit: int = 0


@dataclass(frozen=True)
class Batch:
    """2*rows sequences: concept stories first, matching antagonist stories second."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pooling_weights: torch.Tensor
    pair: torch.Tensor
    bucket: torch.Tensor
    genre: torch.Tensor
    language: torch.Tensor
    generator: torch.Tensor
    tokens: torch.Tensor
    label_hit: torch.Tensor


def pooling_selector(length: int, skip_tokens: int) -> np.ndarray:
    """Uniform weights over story tokens after the first skip_tokens, keeping at least 16."""
    if length <= 0:
        raise ValueError("cannot pool an empty story")
    first = min(skip_tokens, max(0, length - MIN_POOLED_TOKENS))
    selector = np.zeros(length, dtype=np.float32)
    selector[first:] = 1.0 / (length - first)
    return selector


def token_budget_batches(lengths: Sequence[int], budget: int, max_rows: int = 128) -> list[list[int]]:
    """Length-sorted batches where 2 * rows * padded_width stays under the token budget."""
    if budget <= 0 or max_rows <= 0:
        raise ValueError("batch limits must be positive")
    result: list[list[int]] = []
    current: list[int] = []
    widest = 0
    for position in np.argsort(np.asarray(lengths), kind="stable").tolist():
        width = max(widest, int(lengths[position]))
        if current and (2 * (len(current) + 1) * width > budget or len(current) == max_rows):
            result.append(current)
            current, width = [], int(lengths[position])
        current.append(position)
        widest = width
    if current:
        result.append(current)
    return result


def build_batch(
    rows: Sequence[Row],
    wrapping: Callable[[Row, int], Wrapping],
    *,
    pad_token_id: int,
    skip_tokens: int,
) -> Batch:
    if not rows:
        raise ValueError("cannot build an empty batch")
    sequences = []
    for side in (0, 1):
        for row in rows:
            wrap = wrapping(row, side)
            story = row.concept_ids if side == 0 else row.antagonist_ids
            sequences.append((wrap.prefix, story, wrap.suffix))

    width = max(len(prefix) + len(story) + len(suffix) for prefix, story, suffix in sequences)
    input_ids = np.full((len(sequences), width), pad_token_id, dtype=np.int64)
    attention = np.zeros_like(input_ids)
    weights = np.zeros((len(sequences), 1, width), dtype=np.float32)
    for index, (prefix, story, suffix) in enumerate(sequences):
        start = len(prefix)
        end = start + len(story) + len(suffix)
        input_ids[index, :start] = prefix
        input_ids[index, start : start + len(story)] = story
        input_ids[index, start + len(story) : end] = suffix
        attention[index, :end] = 1
        weights[index, 0, start : start + len(story)] = pooling_selector(len(story), skip_tokens)

    return Batch(
        input_ids=torch.from_numpy(input_ids),
        attention_mask=torch.from_numpy(attention),
        pooling_weights=torch.from_numpy(weights),
        pair=torch.tensor([row.pair for row in rows]),
        bucket=torch.tensor([row.bucket for row in rows]),
        genre=torch.tensor([row.genre for row in rows]),
        language=torch.tensor([row.language for row in rows]),
        generator=torch.tensor([row.generator for row in rows]),
        tokens=torch.tensor([[len(row.concept_ids) for row in rows], [len(row.antagonist_ids) for row in rows]]),
        label_hit=torch.tensor([[row.concept_hit for row in rows], [row.antagonist_hit for row in rows]]),
    )


def mean_pool(hidden_state: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Pool [sequence, token, hidden] with [sequence, 1, token] weights -> [sequence, hidden] fp32."""
    if hidden_state.ndim != 3 or weights.shape != (hidden_state.shape[0], 1, hidden_state.shape[1]):
        raise ValueError("hidden states and pooling weights have incompatible shapes")
    return torch.bmm(weights.to(device=hidden_state.device, dtype=hidden_state.dtype), hidden_state).squeeze(1).float()
