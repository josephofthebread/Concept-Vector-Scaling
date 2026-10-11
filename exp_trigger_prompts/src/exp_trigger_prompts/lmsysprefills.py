from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

log = logging.getLogger("lmsysprefills")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("work/lmsys-gap/lmsys_prefills.json"))
    parser.add_argument("--user-words", type=int, nargs=2, default=(8, 400))
    parser.add_argument("--reply-words", type=int, nargs=2, default=(5, 250))
    args = parser.parse_args()

    from datasets import load_dataset

    data = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    umin, umax = args.user_words
    rmin, rmax = args.reply_words
    items: list[dict[str, object]] = []
    scanned = 0
    for row in data:
        scanned += 1
        if row.get("language") != "English":
            continue
        conv = row.get("conversation") or []
        if len(conv) < 2 or conv[0]["role"] != "user" or conv[1]["role"] != "assistant":
            continue
        user = conv[0]["content"].strip()
        reply = conv[1]["content"].strip()
        uw, rw = len(user.split()), len(reply.split())
        if not (umin <= uw <= umax and rmin <= rw <= rmax):
            continue
        items.append(
            {
                "stem": f"lmsys-{len(items):03d}",
                "scenario": "lmsys",
                "outcome": "native",
                "name": row["conversation_id"],
                "conversation": [{"role": "user", "content": user}, {"role": "assistant", "content": reply}],
            }
        )
        if len(items) >= args.n:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(items, ensure_ascii=False, indent=2))
    log.info("scanned %d rows -> kept %d prefills -> %s", scanned, len(items), args.out)


if __name__ == "__main__":
    main()
