from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = ["Qwen/Qwen3-8B", "unsloth/Meta-Llama-3.1-8B-Instruct", "google/gemma-2-9b-it"]
MAXLEN = 256


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B", choices=MODELS)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--cap", type=int, default=200, help="max rows per (pair, language)")
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--stride", type=int, default=2, help="layer stride within the band")
    args = parser.parse_args()

    tag = args.model.split("/")[-1]
    out = Path(f"work/oodrepsb-{tag}")
    out.mkdir(parents=True, exist_ok=True)

    tbl = pq.read_table(
        "data/semantic.parquet",
        columns=["pair_id", "language_id", "genre_id", "generator_id", "concept_text", "antagonist_text"],
    )
    d = tbl.to_pydict()
    by_pair: dict[int, list[int]] = {}
    for i, pid in enumerate(d["pair_id"]):
        by_pair.setdefault(int(pid), []).append(i)
    pairs = sorted(by_pair)[args.shard :: args.shards]

    attn = "eager" if "gemma" in args.model else "sdpa"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=attn, device_map="auto"
    )
    model.eval()
    device = model.device
    nb = model.config.num_hidden_layers
    band = list(range(round(nb / 3), round(2 * nb / 3) + 1, args.stride))
    print(f"{args.model} shard {args.shard}/{args.shards}: {len(pairs)} concepts, band {band}", flush=True)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(li: int) -> Any:
        def hook(_m: Any, _i: Any, o: Any) -> None:
            captured[li] = (o[0] if isinstance(o, tuple) else o).float()

        return hook

    for li in band:
        model.model.layers[li - 1].register_forward_hook(make_hook(li))

    @torch.no_grad()
    def embed(texts: list[str]) -> np.ndarray:
        chunks = []
        for s in range(0, len(texts), args.batch):
            enc = cast(
                "Any",
                tok(texts[s : s + args.batch], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt"),
            ).to(device)
            model(**enc)
            m = enc["attention_mask"].unsqueeze(-1).float()
            denom = m.sum(1).clamp_min(1.0)
            per = []
            for li in band:
                h = captured[li]
                h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                per.append((h * m).sum(1) / denom)
            chunks.append(torch.stack(per, 1).half().cpu().numpy())
        return np.concatenate(chunks, 0)

    rng = np.random.default_rng(0)
    for pi, pid in enumerate(pairs):
        dst = out / f"pair-{pid}.npz"
        if dst.exists():
            continue
        by_lang: dict[int, list[int]] = {}
        for i in by_pair[pid]:
            by_lang.setdefault(int(d["language_id"][i]), []).append(i)
        sel: list[int] = []
        for idxs in by_lang.values():
            take = idxs if len(idxs) <= args.cap else rng.choice(idxs, args.cap, replace=False).tolist()
            sel.extend(int(i) for i in take)
        rp = embed([d["concept_text"][i] for i in sel])
        ra = embed([d["antagonist_text"][i] for i in sel])
        reps = np.concatenate([rp, ra], 0).astype(np.float16)
        lang = [int(d["language_id"][i]) for i in sel]
        genre = [int(d["genre_id"][i]) for i in sel]
        gen = [int(d["generator_id"][i]) for i in sel]
        np.savez_compressed(
            dst,
            reps=reps,
            y=np.array([1] * len(sel) + [0] * len(sel), np.int8),
            lang=np.array(lang + lang, np.int32),
            genre=np.array(genre + genre, np.int32),
            generator=np.array(gen + gen, np.int32),
            layers=np.array(band, np.int32),
        )
        if pi % 20 == 0:
            print(f"  {pi}/{len(pairs)} pair {pid}: {len(sel) * 2} reps x {len(band)} layers", flush=True)
    print(f"=== shard {args.shard} done ===", flush=True)


if __name__ == "__main__":
    main()
