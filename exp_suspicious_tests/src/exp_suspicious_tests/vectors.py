from __future__ import annotations

import torch
from safetensors.torch import load_file

from exp_suspicious_tests.config import CONSTRUCTIONS, GENDATA_VECTORS, PAIRS, VECTOR_TAGS, vector_tag


def load_vectors_for(checkpoint: str, presentation: str, device: str = "cuda") -> dict[str, torch.Tensor]:
    """Return {construction: [layers, pairs, hidden] f32, rows L2-normalized}."""
    tag = vector_tag(checkpoint, presentation)
    directory = GENDATA_VECTORS / VECTOR_TAGS.get(tag, tag)
    out: dict[str, torch.Tensor] = {}
    for construction in CONSTRUCTIONS:
        tensors = load_file(str(directory / f"{construction}.safetensors"))
        matrix = tensors[construction].to(device=device, dtype=torch.float32)
        if matrix.shape[1] != PAIRS:
            raise ValueError(f"{directory}/{construction}: expected {PAIRS} pairs, got {tuple(matrix.shape)}")
        out[construction] = torch.nn.functional.normalize(matrix, dim=-1)
    return out
