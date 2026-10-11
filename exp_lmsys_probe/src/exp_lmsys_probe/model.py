from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from .config import MODEL, VEC_ROOT, VEC_TAG, swept_layers
from .sources import Doc

log = logging.getLogger("model")


def load(model_name: str = MODEL) -> tuple[Any, Any, torch.Tensor, list[int], int]:
    """Return (tokenizer, model, unit vectors [Ls,P,hidden] on cuda, swept layers, n_pairs)."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    depth = config.num_hidden_layers
    layers = swept_layers(depth)
    config.num_hidden_layers = max(layers)
    model = AutoModel.from_pretrained(
        model_name, config=config, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": "cuda"}
    )
    model.eval()
    log.info("model %s: %d of %d blocks, layers %s", model_name, max(layers), depth, layers)

    block = load_file(str(VEC_ROOT / VEC_TAG[model_name] / "diff.safetensors"))["diff"]
    assert block.shape[0] == depth + 1, (block.shape, depth)
    vectors = torch.nn.functional.normalize(torch.stack([block[layer] for layer in layers]).float(), dim=-1).cuda()
    log.info("vectors %s", tuple(vectors.shape))
    return tokenizer, model, vectors, layers, block.shape[1]


def plan_batches(lengths: np.ndarray, batch: int) -> list[list[int]]:
    """Length-sorted, fixed-size chunks so each batch pads little."""
    order = np.argsort(lengths).tolist()
    return [order[i : i + batch] for i in range(0, len(order), batch)]


def collate(docs: list[Doc], pad: int) -> dict[str, Any]:
    """Pad a batch and index its content tokens into the flattened B*W grid, grouped by doc."""
    width = max(len(ids) for _, ids, _ in docs)
    tokens = np.full((len(docs), width), pad, dtype=np.int64)
    attention = np.zeros((len(docs), width), dtype=np.int64)
    flat: list[int] = []
    doc_index: list[int] = []
    clen: list[int] = []
    for row, (_, ids, spans) in enumerate(docs):
        tokens[row, : len(ids)] = ids
        attention[row, : len(ids)] = 1
        base = row * width
        n = 0
        for start, length in spans:
            flat.extend(range(base + start, base + start + length))
            doc_index.extend([row] * length)
            n += length
        clen.append(n)
    return {
        "tokens": torch.from_numpy(tokens).cuda(),
        "attention": torch.from_numpy(attention).cuda(),
        "flat_idx": torch.tensor(flat, device="cuda"),
        "doc_index": torch.tensor(doc_index, dtype=torch.int64, device="cuda"),
        "clen": torch.tensor(clen, dtype=torch.float32, device="cuda"),
        "width": width,
    }


def forward_layers(
    model: Any, tokens: torch.Tensor, attention: torch.Tensor, layers: list[int]
) -> dict[int, torch.Tensor]:
    """Forward once, capturing the swept layers' pre-norm residuals via block hooks -> {L: [B,W,hid]}."""
    ##
    # NOTE: hidden_states[L] is the output of decoder block L-1, so layer L hooks model.layers[L-1].
    # Block hooks fire before the final RMSNorm, matching the diff vectors (hidden_states[-1] is post-norm).
    ##
    captured: dict[int, torch.Tensor] = {}

    def hook(_m: Any, _i: Any, out: Any, layer: int) -> None:
        captured[layer] = out[0] if isinstance(out, tuple) else out

    handles = [
        model.layers[layer - 1].register_forward_hook(lambda m, i, o, layer=layer: hook(m, i, o, layer))
        for layer in layers
    ]
    try:
        model(input_ids=tokens, attention_mask=attention, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return captured


def content_cos(hidden: torch.Tensor, flat_idx: torch.Tensor, vec_layer: torch.Tensor) -> torch.Tensor:
    """Raw cosine of every content token's residual against every unit vector at one layer -> [U, P]."""
    flat = hidden.reshape(-1, hidden.shape[-1])
    unit = torch.nn.functional.normalize(flat.index_select(0, flat_idx).float(), dim=-1)
    return unit @ vec_layer.T
