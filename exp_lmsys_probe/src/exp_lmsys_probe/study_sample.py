from __future__ import annotations

import json
import logging
import random
from argparse import ArgumentParser
from pathlib import Path

log = logging.getLogger("study_sample")

STUDY_N = 16000
ZERO_TOP = 24
COVERED_YES = 3
COVERED_NO = 3
SEED = 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = ArgumentParser()
    parser.add_argument("--work", type=Path, default=Path("work"))
    parser.add_argument("--n", type=int, default=STUDY_N)
    args = parser.parse_args()
    rng = random.Random(SEED)

    judged = {f["pair_id"]: f for f in json.loads((args.work / "judged.json").read_text())["features"]}
    cand = {f["pair_id"]: f for f in json.loads((args.work / "candidates.json").read_text())["features"]}

    slots: list[dict] = []
    for pid, cf in cand.items():
        jf = judged[pid]
        for pole, yes_key in (("con", "concept_yes"), ("ant", "antagonist_yes")):
            verdict = {c["conversation_id"]: c["verdict"] for c in jf[pole]}
            ranked = [
                {
                    "pair_id": pid,
                    "pole": pole,
                    "conversation_id": c["conversation_id"],
                    "layer": c["layer"],
                    "score": c["score"],
                    "rank": r,
                    "plain": verdict.get(c["conversation_id"], "?"),
                    "concept": cf["concept"],
                    "antagonist": cf["antagonist"],
                    "class": cf["class"],
                }
                for r, c in enumerate(cf[pole])
            ]
            if jf[yes_key] == 0:
                for s in ranked[:ZERO_TOP]:
                    s["bucket"] = "recovery"
                    slots.append(s)
            else:
                yes = [s for s in ranked if s["plain"] == "yes"]
                no = [s for s in ranked if s["plain"] == "no"]
                for s in yes[:COVERED_YES]:
                    s["bucket"] = "degradation"
                    slots.append(s)
                for s in no[:: max(1, len(no) // COVERED_NO)][:COVERED_NO] if no else []:
                    s["bucket"] = "baseline"
                    slots.append(s)

    rng.shuffle(slots)
    if len(slots) > args.n:
        slots = slots[: args.n]
    (args.work / "study_slots.json").write_text(json.dumps(slots, ensure_ascii=False))

    from collections import Counter

    by_bucket = Counter(s["bucket"] for s in slots)
    by_pole = Counter(s["pole"] for s in slots)
    by_plain = Counter(s["plain"] for s in slots)
    log.info(
        "%d slots -> study_slots.json | buckets=%s poles=%s plain=%s | %d distinct convs",
        len(slots),
        dict(by_bucket),
        dict(by_pole),
        dict(by_plain),
        len({s["conversation_id"] for s in slots}),
    )


if __name__ == "__main__":
    main()
