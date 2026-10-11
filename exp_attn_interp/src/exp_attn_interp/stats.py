from __future__ import annotations

import torch


def ranks(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    order = values.argsort(dim=dim)
    rank = torch.empty_like(order)
    arange = torch.arange(values.shape[dim], device=values.device).expand_as(order)
    rank.scatter_(dim, order, arange)
    return rank.to(torch.float32)


def spearman_matrix(attention: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
    """attention [heads, T] x activation [T, vectors] -> rho [heads, vectors], Spearman over T."""
    ra = ranks(attention, dim=-1)
    rb = ranks(activation.T, dim=-1)
    ra = ra - ra.mean(dim=-1, keepdim=True)
    rb = rb - rb.mean(dim=-1, keepdim=True)
    num = ra @ rb.T
    den = ra.norm(dim=-1)[:, None] * rb.norm(dim=-1)[None, :]
    return num / den.clamp_min(1e-12)


def rank_auroc(scores: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    """Mann-Whitney AUROC of `positive` (bool [T]) vs the rest per row of `scores` [rows, T]; NaN if a class is empty."""
    n_pos = int(positive.sum().item())
    n_neg = int(positive.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return torch.full((scores.shape[0],), float("nan"), device=scores.device)
    order = scores.argsort(dim=-1)
    rank = torch.empty_like(order, dtype=torch.float32)
    arange = torch.arange(scores.shape[-1], device=scores.device, dtype=torch.float32).expand_as(order)
    rank.scatter_(-1, order, arange)
    rank = rank + 1.0
    pos_rank_sum = rank[:, positive].sum(dim=-1)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def participation_ratio(eigenvalues: torch.Tensor) -> float:
    """(sum lambda)^2 / sum lambda^2 -- effective number of dimensions carrying the variance."""
    positive = eigenvalues.clamp_min(0)
    total = positive.sum()
    if total <= 0:
        return 0.0
    return float(total**2 / (positive**2).sum().clamp_min(1e-30))
