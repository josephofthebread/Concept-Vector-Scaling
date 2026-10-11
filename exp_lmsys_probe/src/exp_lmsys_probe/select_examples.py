from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("select_examples")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--per-feature", type=int, default=16)
    ap.add_argument(
        "--verdict",
        default="yes",
        choices=("yes", "no"),
        help="'yes' = judge-accepted (default); 'no' = judge-rejected",
    )
    ap.add_argument("--out", default="examples_sel.json", help="output filename under --work")
    ap.add_argument("--judged", default="judged.json", help="verdict filename under --work")
    ap.add_argument(
        "--all",
        action="store_true",
        help="sweep selection: every candidate of every pair/pole (top --per-pole by score), "
        "no verdict filter, no distinctness -- input for the span-judge sweep readout",
    )
    ap.add_argument("--per-pole", type=int, default=128)
    args = ap.parse_args()

    cand = {f["pair_id"]: f for f in json.loads((args.work / "candidates.json").read_text())["features"]}
    if args.all:
        feats, total, convs = [], 0, set()
        for pid, cf in cand.items():
            rows = []
            for pole in ("con", "ant"):
                for c in sorted(cf[pole], key=lambda c: -float(c["score"]))[: args.per_pole]:
                    rows.append(
                        {
                            "conversation_id": c["conversation_id"],
                            "pole": pole,
                            "score": round(float(c["score"]), 4),
                            "layer": int(c["layer"]),
                        }
                    )
                    convs.add(c["conversation_id"])
            total += len(rows)
            feats.append(
                {
                    "pair_id": pid,
                    "concept": cf["concept"],
                    "antagonist": cf["antagonist"],
                    "class": cf["class"],
                    "examples": rows,
                }
            )
        (args.work / args.out).write_text(json.dumps({"features": feats}, ensure_ascii=False))
        log.info(
            "all: %d (pair, pole, conv) examples over %d distinct conversations -> %s", total, len(convs), args.out
        )
        return

    judged = {f["pair_id"]: f for f in json.loads((args.work / args.judged).read_text())["features"]}

    best: dict[str, tuple[float, int, str, int]] = {}
    for pid, cf in cand.items():
        jf = judged[pid]
        for pole in ("con", "ant"):
            keep = {c["conversation_id"] for c in jf[pole] if c["verdict"] == args.verdict}
            for c in cf[pole]:
                cid = c["conversation_id"]
                if cid not in keep:
                    continue
                s = float(c["score"])
                if cid not in best or s > best[cid][0]:
                    best[cid] = (s, pid, pole, int(c["layer"]))

    by_feat: dict[int, list[tuple[float, str, str, int]]] = defaultdict(list)
    for cid, (s, pid, pole, layer) in best.items():
        by_feat[pid].append((s, cid, pole, layer))

    half = args.per_feature // 2
    feats = []
    total = 0
    for pid, cf in cand.items():
        assigned = by_feat.get(pid, [])
        con = sorted([r for r in assigned if r[2] == "con"], reverse=True)
        ant = sorted([r for r in assigned if r[2] == "ant"], reverse=True)
        rows = con[:half] + ant[:half]
        if len(rows) < args.per_feature:
            rows += sorted(con[half:] + ant[half:], reverse=True)[: args.per_feature - len(rows)]
        rows.sort(reverse=True)
        total += len(rows)
        feats.append(
            {
                "pair_id": pid,
                "concept": cf["concept"],
                "antagonist": cf["antagonist"],
                "class": cf["class"],
                "examples": [
                    {"conversation_id": cid, "pole": pole, "score": round(s, 4), "layer": layer}
                    for s, cid, pole, layer in rows
                ],
            }
        )

    covered = sum(1 for f in feats if f["examples"])
    (args.work / args.out).write_text(json.dumps({"features": feats}, ensure_ascii=False))
    log.info(
        "verdict=%s: assigned %d distinct conversations | %d selected examples | %d/%d features "
        "with >=1 | median %d/feature -> %s",
        args.verdict,
        len(best),
        total,
        covered,
        len(feats),
        sorted(len(f["examples"]) for f in feats)[len(feats) // 2],
        args.out,
    )


if __name__ == "__main__":
    main()
