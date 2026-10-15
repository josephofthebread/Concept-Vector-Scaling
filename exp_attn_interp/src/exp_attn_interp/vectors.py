from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from exp_attn_interp.config import GENVECTORS_VECTORS, PAIRS, VECTOR_TAG


def load_diff_vectors(device: str = "cuda", root: Path | None = None, tag: str = VECTOR_TAG) -> torch.Tensor:
    """Return [layers, pairs, hidden] float32, rows L2-normalized (layers index 0..n_blocks).
    `tag` selects the exp_genvectors run, e.g. 'unsloth_Meta-Llama-3.1-8B-Instruct--assistant--semantic'."""
    directory = (root or GENVECTORS_VECTORS) / tag
    tensors = load_file(str(directory / "diff.safetensors"))
    matrix = tensors["diff"].to(device=device, dtype=torch.float32)
    if matrix.shape[1] != PAIRS:
        raise ValueError(f"{directory}/diff: expected {PAIRS} pairs, got {tuple(matrix.shape)}")
    return torch.nn.functional.normalize(matrix, dim=-1)
