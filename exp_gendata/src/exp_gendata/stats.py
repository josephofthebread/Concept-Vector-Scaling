from __future__ import annotations

import numpy as np
import torch

from exp_gendata.batching import Batch

SIDES = 2


class SufficientStatistics:
    def __init__(
        self,
        *,
        layers: int,
        pairs: int,
        hidden: int,
        buckets: int,
        genres: int = 10,
        languages: int = 5,
        generators: int = 2,
        device: str | torch.device = "cpu",
        moments: bool = True,
        cells: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.layers, self.pairs, self.hidden, self.buckets = layers, pairs, hidden, buckets
        self.want_moments, self.want_cells = moments, cells

        def zeros(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            return torch.zeros(shape, dtype=dtype, device=self.device)

        self.sums = zeros(layers, pairs, SIDES, buckets, hidden)
        self.corpus_sum = zeros(layers, hidden)
        self.unit_diff_sum = zeros(layers, pairs, buckets, hidden)
        self.counts = zeros(pairs, SIDES, buckets, dtype=torch.int64)
        self.tokens = zeros(pairs, SIDES, buckets, dtype=torch.int64)
        self.label_hits = zeros(pairs, SIDES, buckets, dtype=torch.int64)
        self.moments = zeros(layers, buckets, hidden, hidden, dtype=torch.float64) if moments else None
        self.diff_sq_norm = zeros(layers, pairs, buckets, dtype=torch.float64)
        self.genre_sum = zeros(layers, pairs, SIDES, genres, hidden) if cells else None
        self.language_sum = zeros(layers, pairs, SIDES, languages, hidden)
        self.generator_sum = zeros(layers, pairs, SIDES, generators, hidden) if cells else None
        self.genre_count = zeros(pairs, SIDES, genres, dtype=torch.int64)
        self.language_count = zeros(pairs, SIDES, languages, dtype=torch.int64)
        self.generator_count = zeros(pairs, SIDES, generators, dtype=torch.int64)
        self.nonfinite_rows = 0

    @staticmethod
    def _add(target: torch.Tensor, index: torch.Tensor, values: torch.Tensor) -> None:
        """index_add over every axis but the trailing hidden axis, via a flat view."""
        target.view(-1, target.shape[-1]).index_add_(0, index, values)

    def add(self, pooled: torch.Tensor, batch: Batch) -> None:
        """Accumulate pooled residuals shaped [layer, side, row, hidden]."""
        rows = batch.pair.numel()
        if pooled.shape != (self.layers, SIDES, rows, self.hidden):
            raise ValueError(f"pooled shape {tuple(pooled.shape)} != {(self.layers, SIDES, rows, self.hidden)}")
        finite = torch.isfinite(pooled).all(dim=3).all(dim=1).all(dim=0)
        self.nonfinite_rows += rows - int(finite.sum())
        if not bool(finite.any()):
            return
        keep = finite.cpu()
        pooled = pooled[:, :, finite].to(self.device, dtype=torch.float32)
        pair = batch.pair[keep].to(self.device)
        bucket = batch.bucket[keep].to(self.device)
        rows = pair.numel()
        ones = torch.ones(rows, dtype=torch.int64, device=self.device)
        sides = torch.arange(SIDES, device=self.device)

        cell = (pair[:, None] * SIDES + sides) * self.buckets + bucket[:, None]
        flat_cell = cell.reshape(-1)
        for layer in range(self.layers):
            self._add(self.sums[layer], flat_cell, pooled[layer].movedim(0, 1).reshape(-1, self.hidden))
        self.corpus_sum += pooled.sum(dim=(1, 2))
        self.counts.view(-1).index_add_(0, flat_cell, ones[:, None].expand(-1, SIDES).reshape(-1))
        self.tokens.view(-1).index_add_(0, flat_cell, batch.tokens[:, keep].T.reshape(-1).to(self.device))
        self.label_hits.view(-1).index_add_(0, flat_cell, batch.label_hit[:, keep].T.reshape(-1).to(self.device))

        difference = pooled[:, 0] - pooled[:, 1]
        unit = difference / difference.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        pair_bucket = pair * self.buckets + bucket
        for layer in range(self.layers):
            self._add(self.unit_diff_sum[layer], pair_bucket, unit[layer])
            self.diff_sq_norm[layer].view(-1).index_add_(
                0, pair_bucket, difference[layer].double().square().sum(dim=-1)
            )

        factors = tuple(
            (level.to(self.device), level_sum, level_count)
            for level, level_sum, level_count in (
                (batch.genre[keep], self.genre_sum, self.genre_count),
                (batch.language[keep], self.language_sum, self.language_count),
                (batch.generator[keep], self.generator_sum, self.generator_count),
            )
            if level_sum is not None
        )
        for level, level_sum, level_count in factors:
            levels = level_count.shape[-1]
            level_cell = ((pair[:, None] * SIDES + sides) * levels + level[:, None]).reshape(-1)
            level_count.view(-1).index_add_(0, level_cell, ones[:, None].expand(-1, SIDES).reshape(-1))
            for layer in range(self.layers):
                self._add(level_sum[layer], level_cell, pooled[layer].movedim(0, 1).reshape(-1, self.hidden))

        if self.moments is not None:
            for layer in range(self.layers):
                stories = pooled[layer].reshape(-1, self.hidden)
                story_bucket = bucket.repeat(SIDES)
                for b in range(self.buckets):
                    chosen = stories[story_bucket == b].double()
                    if chosen.numel():
                        self.moments[layer, b].addmm_(chosen.T, chosen)

    def numpy(self) -> dict[str, np.ndarray]:
        names = (
            "sums",
            "corpus_sum",
            "unit_diff_sum",
            "counts",
            "tokens",
            "label_hits",
            "moments",
            "diff_sq_norm",
            "genre_sum",
            "language_sum",
            "generator_sum",
            "genre_count",
            "language_count",
            "generator_count",
        )
        return {name: getattr(self, name).cpu().numpy() for name in names if getattr(self, name) is not None}
