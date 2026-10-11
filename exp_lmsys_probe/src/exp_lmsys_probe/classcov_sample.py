from __future__ import annotations

import json
import logging
import zlib
from argparse import ArgumentParser
from pathlib import Path

from .config import CAP, GARBAGE_FILTER, MIN_TOKENS, MODEL
from .sources import LMSYS, garbage

log = logging.getLogger("classcov_sample")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--work", type=Path, default=Path("work"))
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    stride = max(1, 1_000_000 // args.n)
    root = Path(snapshot_download(LMSYS, repo_type="dataset"))
    out: list[dict] = []
    for shard in sorted(root.rglob("*.parquet")):
        table = pq.read_table(shard, columns=["conversation_id", "conversation"])
        for cid, turns in zip(table.column("conversation_id").to_pylist(), table.column("conversation").to_pylist()):
            if zlib.crc32(cid.encode()) % stride != 0:
                continue
            oi = next((i for i, t in enumerate(turns) if t["role"] == "user"), None)
            if oi is None or not turns[oi]["content"]:
                continue
            reply = next((t["content"] for t in turns[oi + 1 :] if t["role"] == "assistant"), None)
            if not reply:
                continue
            u = tok(turns[oi]["content"], add_special_tokens=False)["input_ids"][:CAP]
            a = tok(reply, add_special_tokens=False)["input_ids"][:CAP]
            if len(u) + len(a) < MIN_TOKENS or not u or not a:
                continue
            if GARBAGE_FILTER and garbage(turns[oi]["content"], u):
                continue
            out.append({"conversation_id": cid, "user": tok.decode(u), "assistant": tok.decode(a)})
            if len(out) >= args.n:
                break
        if len(out) >= args.n:
            break

    (args.work / "classcov_convs.json").write_text(json.dumps(out, ensure_ascii=False))
    log.info("sampled %d conversations -> classcov_convs.json", len(out))


if __name__ == "__main__":
    main()
