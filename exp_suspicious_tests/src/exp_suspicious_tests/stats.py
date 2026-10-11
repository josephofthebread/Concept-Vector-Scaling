from __future__ import annotations

import torch


def ranks(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Ranks by double argsort, ties broken by position."""
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


def spearman_pair(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    a = ranks(a, dim=dim)
    b = ranks(b, dim=dim)
    a = a - a.mean(dim=dim, keepdim=True)
    b = b - b.mean(dim=dim, keepdim=True)
    num = (a * b).sum(dim=dim)
    den = a.norm(dim=dim) * b.norm(dim=dim)
    return num / den.clamp_min(1e-12)
