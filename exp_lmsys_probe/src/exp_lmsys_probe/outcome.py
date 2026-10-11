from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("outcome")
    parser = ArgumentParser()
    parser.add_argument("--work", type=Path, default=Path("work"))
    parser.add_argument("--features", type=Path, default=Path("data/features.json"))
    args = parser.parse_args()

    judged = json.loads((args.work / "judged.json").read_text())["features"]
    feats = json.loads(args.features.read_text())["features"]

    outcome, figman = [], []
    for jf in judged:
        p = jf["pair_id"]
        f = feats[p]
        accepted = [c["conversation_id"] for c in jf["candidates"] if c["verdict"] == "yes"]
        common = {"pair_id": p, "concept": f["concept"], "antagonist": f["antagonist"], "class": f["class_name"]}
        outcome.append({**common, "conversations": accepted})
        figman.append({**common, "top3": accepted[:3]})

    (args.work / "outcome.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=1))
    (args.work / "figman.json").write_text(json.dumps(figman, ensure_ascii=False))
    total = sum(len(o["conversations"]) for o in outcome)
    covered = sum(bool(o["conversations"]) for o in outcome)
    log.info("outcome: %d features, %d covered, %d total accepted conversations", len(outcome), covered, total)


if __name__ == "__main__":
    main()
