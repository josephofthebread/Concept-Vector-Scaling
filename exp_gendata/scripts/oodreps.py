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
    parser.add_argument("--layer", type=int, default=0, help="0 = mid depth")
    parser.add_argument("--cap", type=int, default=80, help="max rows per (pair, language)")
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    tag = args.model.split("/")[-1]
    out = Path(f"work/oodreps-{tag}")
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
    print(f"{args.model} shard {args.shard}/{args.shards}: {len(pairs)} concepts", flush=True)

    attn = "eager" if "gemma" in args.model else "sdpa"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=attn, device_map="auto"
    )
    model.eval()
    device = model.device
    layer = args.layer or round(model.config.num_hidden_layers / 2)
    captured: dict[str, torch.Tensor] = {}

    def hook(_m: Any, _i: Any, o: Any) -> None:
        captured["h"] = (o[0] if isinstance(o, tuple) else o).float()

    model.model.layers[layer - 1].register_forward_hook(hook)

    @torch.no_grad()
    def embed(texts: list[str]) -> np.ndarray:
        reps = []
        for s in range(0, len(texts), args.batch):
            enc = cast(
                "Any",
                tok(texts[s : s + args.batch], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt"),
            ).to(device)
            model(**enc)
            h = captured["h"]
            h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            m = enc["attention_mask"].unsqueeze(-1).float()
            rep = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
            reps.append(rep.half().cpu().numpy())
        return np.concatenate(reps, 0)

    rng = np.random.default_rng(0)
    for pi, pid in enumerate(pairs):
        dst = out / f"pair-{pid}.npz"
        if dst.exists():
            continue
        rows = by_pair[pid]
        by_lang: dict[int, list[int]] = {}
        for i in rows:
            by_lang.setdefault(int(d["language_id"][i]), []).append(i)
        sel: list[int] = []
        for lang, idxs in by_lang.items():
            take = idxs if len(idxs) <= args.cap else rng.choice(idxs, args.cap, replace=False).tolist()
            sel.extend(int(i) for i in take)
        pole_txt = [d["concept_text"][i] for i in sel]
        antg_txt = [d["antagonist_text"][i] for i in sel]
        lang = [int(d["language_id"][i]) for i in sel]
        genre = [int(d["genre_id"][i]) for i in sel]
        gen = [int(d["generator_id"][i]) for i in sel]
        rp = embed(pole_txt)
        ra = embed(antg_txt)
        reps = np.concatenate([rp, ra], 0).astype(np.float16)
        y = np.array([1] * len(sel) + [0] * len(sel), np.int8)
        np.savez_compressed(
            dst,
            reps=reps,
            y=y,
            lang=np.array(lang + lang, np.int32),
            genre=np.array(genre + genre, np.int32),
            generator=np.array(gen + gen, np.int32),
        )
        if pi % 20 == 0:
            print(f"  {pi}/{len(pairs)} pair {pid}: {len(sel) * 2} reps", flush=True)
    print(f"=== shard {args.shard} done ===", flush=True)


if __name__ == "__main__":
    main()
