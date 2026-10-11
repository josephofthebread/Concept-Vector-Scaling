from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

log = logging.getLogger("lmsysclean")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("work/margin/prefills.json"))
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset("OpenLeecher/lmsys_chat_1m_clean", split="train", streaming=True)
    items = []
    scanned = 0
    for row in ds:
        scanned += 1
        conv = row.get("conversations") or []
        if len(conv) < 2 or conv[0]["from"] != "human" or conv[1]["from"] != "gpt":
            continue
        user = conv[0]["value"].strip()
        reply = conv[1]["value"].strip()
        if (
            sum(c.isascii() for c in user) / max(len(user), 1) < 0.95
            or sum(c.isascii() for c in reply) / max(len(reply), 1) < 0.95
        ):
            continue
        uw, rw = len(user.split()), len(reply.split())
        if not (8 <= uw <= 300 and 8 <= rw <= 300):
            continue
        items.append(
            {
                "stem": f"clean-{len(items):03d}",
                "conversation": [{"role": "user", "content": user}, {"role": "assistant", "content": reply}],
            }
        )
        if len(items) >= args.n:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(items, ensure_ascii=False))
    log.info("scanned %d -> kept %d english prefills -> %s", scanned, len(items), args.out)


if __name__ == "__main__":
    main()
