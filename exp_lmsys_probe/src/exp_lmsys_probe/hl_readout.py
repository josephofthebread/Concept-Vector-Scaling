from __future__ import annotations

import json
import logging
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .config import CAP, MODEL, batch_for
from .model import collate, forward_layers, load, plan_batches
from .sources import iter_docs

log = logging.getLogger("hl_readout")


def combine(work: Path, sel_name: str, prefix: str, merged: str, compact: bool) -> None:
    sel = json.loads((work / sel_name).read_text())["features"]
    if not compact:
        hl: dict[str, dict] = {}
        for f in sorted(work.glob(f"{prefix}-*.json")):
            for r in json.loads(f.read_text()):
                hl[r["conversation_id"]] = r
        out = []
        for f in sel:
            exs = []
            for e in f["examples"]:
                r = hl.get(e["conversation_id"])
                if r is None:
                    continue
                exs.append(
                    {
                        "conversation_id": e["conversation_id"],
                        "pole": e["pole"],
                        "score": e["score"],
                        "layer": e["layer"],
                        "ncut": r["ncut"],
                        "toks": r["toks"],
                        "cos": r["cos"],
                    }
                )
            out.append({k: f[k] for k in ("pair_id", "concept", "antagonist", "class")} | {"examples": exs})
        (work / merged).write_text(json.dumps({"features": out}, ensure_ascii=False))
        log.info(
            "combined -> %s (%d examples over %d features)", merged, sum(len(f["examples"]) for f in out), len(out)
        )
        return

    convs: dict[str, dict] = {}
    cos: dict[tuple[str, int], list[float]] = {}
    for f in sorted(work.glob(f"{prefix}-*.json")):
        d = json.loads(f.read_text())
        convs.update(d["convs"])
        for r in d["rows"]:
            cos[(r["conversation_id"], r["pair_id"])] = r["cos"]
    out = []
    for f in sel:
        exs = []
        for e in f["examples"]:
            c = cos.get((e["conversation_id"], f["pair_id"]))
            if c is None:
                continue
            exs.append(
                {
                    "conversation_id": e["conversation_id"],
                    "pole": e["pole"],
                    "score": e["score"],
                    "layer": e["layer"],
                    "cos": c,
                }
            )
        out.append({k: f[k] for k in ("pair_id", "concept", "antagonist", "class")} | {"examples": exs})
    (work / merged).write_text(json.dumps({"features": out, "convs": convs}, ensure_ascii=False))
    log.info(
        "combined -> %s (%d examples over %d features, %d conversations)",
        merged,
        sum(len(f["examples"]) for f in out),
        len(out),
        len(convs),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--corpus", default="lmsys")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--sel", default="examples_sel.json", help="selection file under --work")
    ap.add_argument("--prefix", default="hl", help="per-shard output prefix ({prefix}-i.json)")
    ap.add_argument("--merged", default="highlights.json", help="merged output filename")
    ap.add_argument(
        "--compact",
        action="store_true",
        help="sweep format: tokens once per conversation, cos per (conversation, pair)",
    )
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="count the shard's work, no model")
    args = ap.parse_args()
    if args.merge:
        combine(args.work, args.sel, args.prefix, args.merged, args.compact)
        return
    started = time.monotonic()

    sel = json.loads((args.work / args.sel).read_text())["features"]
    want: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for f in sel:
        for e in f["examples"]:
            want[e["conversation_id"]].append((f["pair_id"], int(e["layer"])))
    mine = set(sorted(want)[args.shard :: args.shards])
    n_rows = sum(len(want[c]) for c in mine)
    log.info(
        "shard %d/%d: %d of %d conversations, %d (conversation, pair) rows",
        args.shard,
        args.shards,
        len(mine),
        len(want),
        n_rows,
    )
    if args.dry_run:
        return

    tokenizer, model, vectors, layers, _ = load(args.model)
    lidx = {layer: i for i, layer in enumerate(layers)}
    pad = tokenizer.pad_token_id or 0

    docs = []
    seen: set[str] = set()
    for doc in iter_docs(args.corpus, tokenizer, CAP):
        if doc[0] in mine and doc[0] not in seen:
            docs.append(doc)
            seen.add(doc[0])
            if len(seen) == len(mine):
                break
    log.info(
        "collected %d/%d conversations from %s (%.0fs)", len(docs), len(mine), args.corpus, time.monotonic() - started
    )
    lengths = np.asarray([len(x) for _, x, _ in docs], dtype=np.int64)

    out: list[dict] = []
    rows: list[dict] = []
    convs: dict[str, dict] = {}
    plan = plan_batches(lengths, batch_for(torch.cuda.get_device_name(0)))
    with torch.inference_mode():
        for step, batch in enumerate(plan):
            bdocs = [docs[i] for i in batch]
            packed = collate(bdocs, pad)
            need = sorted({layer for d in bdocs for _, layer in want[d[0]]})
            hs = forward_layers(model, packed["tokens"], packed["attention"], need)
            for r, (cid, ids, spans) in enumerate(bdocs):
                pos = [p for (start, length) in spans for p in range(start, start + length)]
                ncut = spans[0][1] if len(spans) > 1 else 0
                toks = [tokenizer.decode([ids[p]]) for p in pos]
                by_layer: dict[int, list[int]] = defaultdict(list)
                for pid, layer in want[cid]:
                    by_layer[layer].append(pid)
                for layer, pids in by_layer.items():
                    unit = torch.nn.functional.normalize(hs[layer][r, pos].float(), dim=-1)
                    cos = (unit @ vectors[lidx[layer], pids].T).cpu().numpy()
                    for j, pid in enumerate(pids):
                        c = [round(float(x), 4) for x in cos[:, j]]
                        if args.compact:
                            rows.append({"conversation_id": cid, "pair_id": pid, "cos": c})
                        else:
                            out.append({"conversation_id": cid, "ncut": ncut, "toks": toks, "cos": c})
                if args.compact:
                    convs[cid] = {"ncut": ncut, "toks": toks}
            if step % 10 == 0 or step + 1 == len(plan):
                log.info("batch %d/%d (%.0fs)", step + 1, len(plan), time.monotonic() - started)

    args.work.mkdir(parents=True, exist_ok=True)
    path = args.work / f"{args.prefix}-{args.shard}.json"
    path.write_text(json.dumps({"convs": convs, "rows": rows} if args.compact else out, ensure_ascii=False))
    log.info(
        "shard %d done in %.0fs: %d rows over %d conversations -> %s",
        args.shard,
        time.monotonic() - started,
        len(rows) if args.compact else len(out),
        len(docs),
        path.name,
    )


if __name__ == "__main__":
    main()
