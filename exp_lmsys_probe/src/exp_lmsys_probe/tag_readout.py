from __future__ import annotations

import json
import logging
import re
import time
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import CAP, MODEL, batch_for
from .model import collate, forward_layers, load, plan_batches
from .sources import iter_docs

log = logging.getLogger("tag_readout")
_SPACES = re.compile(r" +")


def group_words(ids: list[int], tok: Any) -> list[list[int]]:
    """Group content-token indices into words: a token continues the previous word only if it has no
    leading space and is an ASCII alphanumeric subword (so CJK and punctuation stay their own units)."""
    words: list[list[int]] = []
    for i, tid in enumerate(ids):
        p = tok.decode([tid])
        cont = i > 0 and p[:1] not in (" ", "\n", "\t", "\r") and p.isascii() and p.strip().isalnum()
        if cont and words:
            words[-1].append(i)
        else:
            words.append([i])
    return words


def emit(ids: list[int], marks: list[str], tok: Any) -> str:
    """Decode content tokens, wrapping contiguous same-mark runs: 'c' -> << >>, 'a' -> [[ ]]."""
    out: list[str] = []
    i = 0
    while i < len(ids):
        j = i
        while j < len(ids) and marks[j] == marks[i]:
            j += 1
        seg = tok.decode(ids[i:j]).strip()
        if seg:
            out.append(f" <<{seg}>>" if marks[i] == "c" else f" [[{seg}]]" if marks[i] == "a" else " " + seg)
        i = j
    return _SPACES.sub(" ", "".join(out)).strip()


def word_marks(cont_ids: list[int], cos: np.ndarray, ncut: int, tok: Any, topk: int) -> tuple[list[str], int, int]:
    """Per-token marks: the top-K words over both turns -> 'c', the bottom-K -> 'a'."""
    wu, wa = group_words(cont_ids[:ncut], tok), group_words(cont_ids[ncut:], tok)
    scores = np.array([float(cos[:ncut][w].mean()) for w in wu] + [float(cos[ncut:][w].mean()) for w in wa])
    n = len(scores)
    k = max(1, min(topk, n // 2))
    order = np.argsort(scores)
    wm = [""] * n
    for t in order[:k]:
        wm[t] = "a"
    for t in order[-k:]:
        wm[t] = "c"
    marks = [""] * len(cont_ids)
    for wi, w in enumerate(wu):
        for t in w:
            marks[t] = wm[wi]
    for wi, w in enumerate(wa):
        for t in w:
            marks[ncut + t] = wm[len(wu) + wi]
    return marks, wm.count("c"), wm.count("a")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--work", type=Path, default=Path("work"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--topk", type=int, default=4, help="strongest words marked per pole")
    parser.add_argument("--control-frac", type=float, default=0.10)
    args = parser.parse_args()
    started = time.monotonic()
    rng = np.random.default_rng(args.shard)

    slots = json.loads((args.work / "study_slots.json").read_text())
    all_convs = sorted({s["conversation_id"] for s in slots})
    my_convs = set(all_convs[args.shard :: args.shards])
    my_slots = [s for s in slots if s["conversation_id"] in my_convs]
    by_conv: dict[str, list[dict]] = defaultdict(list)
    for s in my_slots:
        by_conv[s["conversation_id"]].append(s)
    log.info("shard %d/%d: %d slots over %d conversations", args.shard, args.shards, len(my_slots), len(my_convs))

    tokenizer, model, vectors, layers, _ = load(args.model)
    lidx = {layer: i for i, layer in enumerate(layers)}
    pad = tokenizer.pad_token_id or 0

    docs = []
    seen: set[str] = set()
    for doc in iter_docs("lmsys", tokenizer, CAP):
        if doc[0] in my_convs and doc[0] not in seen:
            docs.append(doc)
            seen.add(doc[0])
            if len(seen) == len(my_convs):
                break
    log.info("collected %d/%d conversations from the corpus", len(docs), len(my_convs))
    lengths = np.asarray([len(x) for _, x, _ in docs], dtype=np.int64)

    out: list[dict] = []
    plan = plan_batches(lengths, batch_for(torch.cuda.get_device_name(0)))
    with torch.inference_mode():
        for step, batch in enumerate(plan):
            bdocs = [docs[i] for i in batch]
            packed = collate(bdocs, pad)
            need = sorted({s["layer"] for d in bdocs for s in by_conv[d[0]]})
            hs = forward_layers(model, packed["tokens"], packed["attention"], need)
            for r, (cid, _ids, spans) in enumerate(bdocs):
                pos = [p for (start, length) in spans for p in range(start, start + length)]
                ncut = spans[0][1]
                cont_ids = [_ids[p] for p in pos]
                for s in by_conv[cid]:
                    row = hs[s["layer"]][r, pos].float()
                    unit = torch.nn.functional.normalize(row, dim=-1)
                    cos = (unit @ vectors[lidx[s["layer"]], s["pair_id"]]).cpu().numpy()
                    marks, n_c, n_a = word_marks(cont_ids, cos, ncut, tokenizer, args.topk)
                    rec = {
                        **{
                            k: s[k]
                            for k in (
                                "pair_id",
                                "pole",
                                "conversation_id",
                                "layer",
                                "plain",
                                "bucket",
                                "concept",
                                "antagonist",
                                "class",
                            )
                        },
                        "tagged_user": emit(cont_ids[:ncut], marks[:ncut], tokenizer),
                        "tagged_assistant": emit(cont_ids[ncut:], marks[ncut:], tokenizer),
                        "control": False,
                    }
                    out.append(rec)
                    if rng.random() < args.control_frac:
                        words = [(0, w) for w in group_words(cont_ids[:ncut], tokenizer)] + [
                            (ncut, w) for w in group_words(cont_ids[ncut:], tokenizer)
                        ]
                        fm = [""] * len(cont_ids)
                        for j, wi in enumerate(rng.choice(len(words), size=min(n_c + n_a, len(words)), replace=False)):
                            off, w = words[wi]
                            for t in w:
                                fm[off + t] = "c" if j < n_c else "a"
                        out.append(
                            {
                                **rec,
                                "control": True,
                                "tagged_user": emit(cont_ids[:ncut], fm[:ncut], tokenizer),
                                "tagged_assistant": emit(cont_ids[ncut:], fm[ncut:], tokenizer),
                            }
                        )
            if step % 10 == 0 or step + 1 == len(plan):
                log.info("batch %d/%d (%.0fs)", step + 1, len(plan), time.monotonic() - started)

    missing = len(my_slots) - sum(not r["control"] for r in out)
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / f"study_tagged-{args.shard}.json").write_text(json.dumps(out, ensure_ascii=False))
    log.info(
        "shard %d done in %.0fs: %d records (%d controls), %d slots missing a conversation -> %s",
        args.shard,
        time.monotonic() - started,
        len(out),
        sum(r["control"] for r in out),
        missing,
        f"study_tagged-{args.shard}.json",
    )


if __name__ == "__main__":
    main()
