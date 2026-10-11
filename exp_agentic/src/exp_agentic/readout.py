#! /usr/bin/env python
import argparse
import json
import logging
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file

from exp_agentic.model import load

log = logging.getLogger("readout")

VECTORS = Path("../exp_genvectors/work/vectors/Qwen_Qwen3-8B--assistant--semantic")
LAYERS = {18: 18, 25: 25}
CHUNK = 2048


def find(name: str) -> str:
    """Locate a file of the exp_genvectors Qwen3-8B vectors."""
    path = VECTORS / name
    if not path.is_file():
        raise SystemExit(f"{path} is missing; build the vectors with exp_genvectors first")
    return str(path)


def capture(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor | tuple[torch.Tensor, ...],
    slot: int,
    vectors: torch.Tensor,
    into: dict[int, torch.Tensor],
) -> None:
    """Forward hook projecting one block's residual stream onto every direction."""
    state = (output[0] if isinstance(output, tuple) else output).float()
    if not torch.isfinite(state).all():
        raise RuntimeError("non-finite residual stream during readout; fp16 has overflowed")
    norm = torch.linalg.vector_norm(state, dim=-1, keepdim=True).clamp_min(1e-6)
    into[slot] = ((state / norm) @ vectors.T)[0]
    into[-1 - slot] = norm[0, :, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("episodes/stage1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--redo", action="store_true", help="rescore episodes that already have scores")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    block = load_file(find("diff.safetensors"))["diff"]
    rows = pq.read_table(find("pairs.parquet")).to_pylist()
    names = [f"{row['concept']} || {row['antagonist']}" for row in rows]
    classes = [row["class_name"] for row in rows]

    vectors = torch.nn.functional.normalize(
        torch.stack([block[index] for index in LAYERS.values()]).float(), dim=-1
    ).to(args.device)
    log.info(f"vectors {tuple(vectors.shape)} at layers {list(LAYERS)}")

    model, _ = load(device=args.device, dtype=args.dtype)

    for offset, record in enumerate(sorted(args.dir.glob("*.json"))):
        if offset % args.shards != args.shard:
            continue
        target = record.with_suffix(".scores.npy")
        if target.exists() and not args.redo:
            continue
        episode = json.loads(record.read_text())
        ids = episode["ids"]
        blocks = model.model.layers
        chunk_scores: dict[int, torch.Tensor] = {}
        handles = [
            blocks[layer - 1].register_forward_hook(
                partial(capture, slot=slot, vectors=vectors[slot], into=chunk_scores)
            )
            for slot, layer in enumerate(LAYERS)
        ]
        collected: list[np.ndarray] = []
        norms: list[np.ndarray] = []
        try:
            past, device = None, model.device
            with torch.inference_mode():
                for start in range(0, len(ids), CHUNK):
                    piece = ids[start : start + CHUNK]
                    output = model.model(
                        input_ids=torch.tensor([piece], device=device),
                        attention_mask=torch.ones(1, start + len(piece), dtype=torch.long, device=device),
                        cache_position=torch.arange(start, start + len(piece), device=device),
                        past_key_values=past,
                        use_cache=True,
                    )
                    past = output.past_key_values
                    collected.append(
                        torch.stack([chunk_scores[slot] for slot in range(len(LAYERS))], dim=1).half().cpu().numpy()
                    )
                    norms.append(
                        torch.stack([chunk_scores[-1 - slot] for slot in range(len(LAYERS))], dim=1)
                        .float()
                        .cpu()
                        .numpy()
                    )
        finally:
            for handle in handles:
                handle.remove()
        scores, magnitude = np.concatenate(collected, axis=0), np.concatenate(norms, axis=0)
        np.save(target, scores)
        np.save(record.with_suffix(".norms.npy"), magnitude.astype(np.float32))

        roles = np.array(episode["roles"])
        thinking = scores[roles == "thinking"] if (roles == "thinking").any() else scores
        strength = np.abs(thinking[:, 1, :].astype(np.float32)).mean(axis=0)
        top = np.argsort(-strength)[:5]
        log.info(f"{record.stem}: {scores.shape} -> {target.name}")
        for index in top:
            log.info(f"    L25 {strength[index]:.4f}  {names[index][:58]}  [{classes[index][:28]}]")


if __name__ == "__main__":
    main()
