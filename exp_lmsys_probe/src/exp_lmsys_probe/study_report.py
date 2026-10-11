from __future__ import annotations

import json
import logging
import random
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

log = logging.getLogger("study_report")


def accepts(pole: str, hl: str) -> bool:
    return (pole == "con" and hl == "concept") or (pole == "ant" and hl == "antagonist")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--work", type=Path, default=Path("work"))
    parser.add_argument("--sample", type=int, default=250)
    args = parser.parse_args()

    judged = [r for f in sorted(args.work.glob("study_judged-*.json")) for r in json.loads(f.read_text())]
    tagged = {
        (r["conversation_id"], r["pair_id"], r["control"]): r
        for f in sorted(args.work.glob("study_tagged-*.json"))
        for r in json.loads(f.read_text())
    }
    real = [r for r in judged if not r["control"]]
    ctrl = [r for r in judged if r["control"]]
    log.info("=== %d judged records: %d real, %d false-marking controls ===", len(judged), len(real), len(ctrl))

    dist = Counter(r["hl"] for r in real)
    log.info("highlighted verdict distribution (real): %s", dict(dist))
    top = max(dist.values()) / max(1, len(real))
    log.info("  most-common class share: %.1f%%  (degenerate if >90%%)", 100 * top)

    if ctrl:
        stamped = sum(r["hl"] != "incidental" for r in ctrl)
        log.info(
            "BIAS  false-marking rubber-stamp rate: %d/%d = %.1f%%  (want <15%%)",
            stamped,
            len(ctrl),
            100 * stamped / len(ctrl),
        )

    yes = [r for r in real if r["plain"] == "yes"]
    no = [r for r in real if r["plain"] == "no"]
    if yes:
        kept = sum(accepts(r["pole"], r["hl"]) for r in yes)
        log.info(
            "DEGRAD plain=yes still accepted by highlighted: %d/%d = %.1f%%  (want >80%%)",
            kept,
            len(yes),
            100 * kept / len(yes),
        )
        log.info(
            "       of the rest: %d -> incidental, %d -> opposite pole",
            sum(r["hl"] == "incidental" for r in yes),
            sum(
                (r["pole"] == "con" and r["hl"] == "antagonist") or (r["pole"] == "ant" and r["hl"] == "concept")
                for r in yes
            ),
        )
    if no:
        rec = sum(accepts(r["pole"], r["hl"]) for r in no)
        log.info(
            "RECOV  plain=no now accepted by highlighted: %d/%d = %.1f%%  (the coverage gain)",
            rec,
            len(no),
            100 * rec / len(no),
        )

    plain_cov, hl_cov, seen = set(), set(), set()
    for r in real:
        key = (r["pair_id"], r["pole"])
        seen.add(key)
        if r["plain"] == "yes":
            plain_cov.add(key)
        if accepts(r["pole"], r["hl"]):
            hl_cov.add(key)
    log.info(
        "COVERAGE over %d sampled (feature,pole) cells: plain %d, highlighted %d, gained %d, lost %d",
        len(seen),
        len(plain_cov),
        len(hl_cov),
        len(hl_cov - plain_cov),
        len(plain_cov - hl_cov),
    )

    disagree = [r for r in real if (r["plain"] == "yes") != accepts(r["pole"], r["hl"])]
    random.Random(0).shuffle(disagree)
    dump = []
    for r in disagree[: args.sample]:
        t = tagged.get((r["conversation_id"], r["pair_id"], False), {})
        dump.append(
            {
                **{k: r[k] for k in ("pair_id", "pole", "conversation_id", "plain", "hl", "bucket")},
                "concept": t.get("concept"),
                "antagonist": t.get("antagonist"),
                "tagged_user": t.get("tagged_user"),
                "tagged_assistant": t.get("tagged_assistant"),
                "reason": r["reason"],
            }
        )
    (args.work / "study_disagreements.json").write_text(json.dumps(dump, ensure_ascii=False, indent=1))
    log.info("wrote %d/%d disagreements -> study_disagreements.json (for blind labeling)", len(dump), len(disagree))


if __name__ == "__main__":
    main()
