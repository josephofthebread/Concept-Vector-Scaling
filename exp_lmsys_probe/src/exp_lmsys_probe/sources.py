from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from .config import GARBAGE_FILTER, MIN_ALPHA_FRAC, MIN_TOKENS, MIN_UNIQUE_RATIO

log = logging.getLogger("sources")

Doc = tuple[str, list[int], list[tuple[int, int]]]

LMSYS = "lmsys/lmsys-chat-1m"

_RUN = re.compile(r"(.)\1{29,}")


def garbage(text: str, piece: list[int]) -> bool:
    """True if the user turn is repetition/spam or non-natural-language (language-agnostic)."""
    if len(set(piece)) / len(piece) < MIN_UNIQUE_RATIO:
        return True
    if _RUN.search(text):
        return True
    return sum(c.isalpha() for c in text) < MIN_ALPHA_FRAC * max(len(text), 1)


def exchange_template(tokenizer: Any) -> tuple[list[int], list[int], list[int]]:
    """Head / middle / tail token ids around the user and assistant contents, exact by construction."""
    su, sa = "\x00USER\x00", "\x00ASSISTANT\x00"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": su}, {"role": "assistant", "content": sa}], tokenize=False, enable_thinking=False
    )
    head, rest = rendered.split(su)
    middle, tail = rest.split(sa)
    ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in (head, middle, tail)]
    log.info("template: head %d, middle %d, tail %d tokens", len(ids[0]), len(ids[1]), len(ids[2]))
    return ids[0], ids[1], ids[2]


def lmsys_docs(tokenizer: Any, cap: int) -> Iterator[Doc]:
    """First user turn + the assistant reply it drew; both are scored content."""
    root = Path(snapshot_download(LMSYS, repo_type="dataset"))
    head, middle, tail = exchange_template(tokenizer)
    for shard in sorted(root.rglob("*.parquet")):
        table = pq.read_table(shard, columns=["conversation_id", "conversation"])
        for identifier, turns in zip(
            table.column("conversation_id").to_pylist(), table.column("conversation").to_pylist()
        ):
            oi = next((i for i, t in enumerate(turns) if t["role"] == "user"), None)
            if oi is None or not turns[oi]["content"]:
                continue
            reply = next((t["content"] for t in turns[oi + 1 :] if t["role"] == "assistant"), None)
            if not reply:
                continue
            u = tokenizer(turns[oi]["content"], add_special_tokens=False)["input_ids"][:cap]
            a = tokenizer(reply, add_special_tokens=False)["input_ids"][:cap]
            if len(u) + len(a) < MIN_TOKENS or not u or not a:
                continue
            if GARBAGE_FILTER and garbage(turns[oi]["content"], u):
                continue
            ids = head + u + middle + a + tail
            uspan = (len(head), len(u))
            aspan = (len(head) + len(u) + len(middle), len(a))
            yield identifier, ids, [uspan, aspan]


OPENWEBTEXT = "Skylion007/openwebtext"


def openwebtext_docs(tokenizer: Any, cap: int) -> Iterator[Doc]:
    """Raw pretraining documents, first `cap` tokens each, no chat template.
    A BOS is prepended when the tokenizer has one and excluded from content."""
    from datasets import load_dataset

    bos = tokenizer.bos_token_id
    ds = load_dataset(OPENWEBTEXT, split="train")
    for i, ex in enumerate(ds):
        toks = tokenizer(ex["text"][: cap * 8], add_special_tokens=False)["input_ids"][:cap]
        if len(toks) < MIN_TOKENS:
            continue
        ids = ([bos] + toks) if bos is not None else list(toks)
        cstart = 1 if bos is not None else 0
        yield f"owt-{i}", ids, [(cstart, len(toks))]


ALPACA = "tatsu-lab/alpaca"


def alpaca_user(instruction: str, inp: str) -> str:
    """The user turn = instruction, plus the optional input block when present."""
    return instruction + ("\n\n" + inp if inp and inp.strip() else "")


def alpaca_docs(tokenizer: Any, cap: int) -> Iterator[Doc]:
    """Instruction(+input) as the user turn, the response as the assistant reply; both scored."""
    from datasets import load_dataset

    head, middle, tail = exchange_template(tokenizer)
    ds = load_dataset(ALPACA, split="train")
    for i, ex in enumerate(ds):
        um, out = alpaca_user(ex["instruction"], ex["input"]), ex["output"]
        if not um or not out:
            continue
        u = tokenizer(um, add_special_tokens=False)["input_ids"][:cap]
        a = tokenizer(out, add_special_tokens=False)["input_ids"][:cap]
        if len(u) + len(a) < MIN_TOKENS or not u or not a:
            continue
        if GARBAGE_FILTER and garbage(um, u):
            continue
        ids = head + u + middle + a + tail
        uspan = (len(head), len(u))
        aspan = (len(head) + len(u) + len(middle), len(a))
        yield f"alpaca-{i}", ids, [uspan, aspan]


def iter_docs(corpus: str, tokenizer: Any, cap: int) -> Iterator[Doc]:
    if corpus == "lmsys":
        return lmsys_docs(tokenizer, cap)
    if corpus == "openwebtext":
        return openwebtext_docs(tokenizer, cap)
    if corpus == "alpaca":
        return alpaca_docs(tokenizer, cap)
    raise SystemExit(f"unknown corpus: {corpus}")
