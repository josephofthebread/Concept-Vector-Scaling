from __future__ import annotations

import json
import logging
import time
import zlib
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from .config import CAP, LEVELS, LOWER, MODEL, N_ROWS, POLE_WEIGHTS, TOPK, UPPER, batch_for
from .model import collate, content_cos, forward_layers, load, plan_batches
from .sources import iter_docs

log = logging.getLogger("readout")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--corpus", default="lmsys")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--rows", type=int, default=N_ROWS)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("work"))
    args = parser.parse_args()
    started = time.monotonic()

    tokenizer, model, vectors, layers, pairs = load(args.model)
    n_l, n_lvl, k = len(layers), len(LEVELS), TOPK
    pad = tokenizer.pad_token_id or 0
    levels_t = torch.tensor(LEVELS, device="cuda")
    w = torch.tensor(POLE_WEIGHTS, device="cuda")
    up = torch.tensor(UPPER, device="cuda")
    lo = torch.tensor(LOWER, device="cuda")
    layers_t = torch.tensor(layers, device="cuda")

    stride = max(1, 1_000_000 // args.rows)
    docs = []
    for doc in iter_docs(args.corpus, tokenizer, CAP):
        if zlib.crc32(doc[0].encode()) % stride == 0:
            docs.append(doc)
            if len(docs) >= args.rows:
                break
    docs = docs[args.shard :: args.shards]
    if not docs:
        raise SystemExit("no conversations in this shard")
    ids = [d[0] for d in docs]
    lengths = np.asarray([len(x) for _, x, _ in docs], dtype=np.int64)
    log.info(
        "shard %d/%d: %d conversations, %.2fM content tokens",
        args.shard,
        args.shards,
        len(docs),
        sum(ln for _, _, sp in docs for _, ln in sp) / 1e6,
    )

    def pole() -> dict[str, torch.Tensor]:
        return {
            "score": torch.full((k, pairs), float("-inf"), device="cuda"),
            "conv": torch.full((k, pairs), -1, dtype=torch.int64, device="cuda"),
            "layer": torch.full((k, pairs), -1, dtype=torch.int64, device="cuda"),
            "Q": torch.zeros(k, pairs, n_lvl, device="cuda"),
        }

    con, ant = pole(), pole()

    plan = plan_batches(lengths, batch_for(torch.cuda.get_device_name(0)))
    log.info("GPU %s; %d batches; top-%d/feature per pole", torch.cuda.get_device_name(0), len(plan), k)
    with torch.inference_mode():
        for step, batch in enumerate(plan):
            packed = collate([docs[i] for i in batch], pad)
            clen = packed["clen"]
            b = len(batch)
            offs = np.concatenate([[0], np.cumsum(clen.cpu().numpy().astype(np.int64))])
            hs = forward_layers(model, packed["tokens"], packed["attention"], layers)
            Qb = torch.zeros(b, n_l, pairs, n_lvl, device="cuda")
            for li, layer in enumerate(layers):
                cos = content_cos(hs[layer], packed["flat_idx"], vectors[li])
                for row in range(b):
                    Qb[row, li] = torch.quantile(cos[int(offs[row]) : int(offs[row + 1])], levels_t, dim=0).T
            conv_b = torch.tensor(batch, device="cuda")[:, None].expand(b, pairs)
            for store, s in ((con, (Qb[..., up] * w).sum(-1)), (ant, -(Qb[..., lo] * w).sum(-1))):
                best, li_idx = s.max(dim=1)
                Qbest = Qb.gather(1, li_idx[:, None, :, None].expand(b, 1, pairs, n_lvl))[:, 0]
                payloads = [(store["conv"], conv_b), (store["layer"], layers_t[li_idx]), (store["Q"], Qbest)]
                nv, ni = torch.cat([store["score"], best], 0).topk(k, dim=0)
                store["score"].copy_(nv)
                for g, new in payloads:
                    allp = torch.cat([g, new], 0)
                    idx = ni if allp.dim() == 2 else ni.unsqueeze(-1).expand(-1, -1, allp.shape[-1])
                    g.copy_(allp.gather(0, idx))
            if step % 20 == 0 or step + 1 == len(plan):
                log.info("batch %d/%d (%.0fs)", step + 1, len(plan), time.monotonic() - started)

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(
        {f"{p}_{f}": s[f].cpu() for p, s in (("con", con), ("ant", ant)) for f in s},
        str(args.out / f"cand-{args.shard}.safetensors"),
    )
    (args.out / f"ids-{args.shard}.json").write_text(
        json.dumps(
            {
                "ids": ids,
                "layers": layers,
                "levels": LEVELS,
                "pairs": pairs,
                "topk": k,
                "model": model.name_or_path,
                "corpus": args.corpus,
                "shard": args.shard,
                "shards": args.shards,
            },
            ensure_ascii=False,
        )
    )
    log.info("shard %d done in %.0fs -> %s", args.shard, time.monotonic() - started, args.out)


if __name__ == "__main__":
    main()
