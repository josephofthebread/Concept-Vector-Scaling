#! /usr/bin/env python


import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from exp_agentic.readout import find

log = logging.getLogger("trace")

LAYERS = {18: 0, 25: 1}
GENERATED = ("thinking", "tool_call", "answer")
GROUPS = {"submit": "reward hacked", "give_up": "gave up", "degenerate": "degenerate"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("episodes/gate"))
    parser.add_argument("--pair", type=int, required=True)
    parser.add_argument("--layer", type=int, default=25, choices=[18, 25])
    parser.add_argument("--out", type=Path, default=Path("analysis"))
    parser.add_argument("--max-turns", type=int, default=20, help="turn positions to plot")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    slot = LAYERS[args.layer]

    rows = pq.read_table(find("pairs.parquet")).to_pylist()
    name = f"{rows[args.pair]['concept']} || {rows[args.pair]['antagonist']}"
    klass = rows[args.pair]["class_name"]

    forward: dict[str, dict[int, list[float]]] = {g: defaultdict(list) for g in GROUPS}
    backward: dict[str, dict[int, list[float]]] = {g: defaultdict(list) for g in GROUPS}
    counts: dict[str, int] = defaultdict(int)

    for path in sorted(args.dir.glob("*.z.npy")):
        record = Path(str(path).replace(".z.npy", ".json"))
        episode = json.loads(record.read_text())
        ending = episode.get("ending")
        if ending not in GROUPS:
            continue
        z = np.load(path)
        roles = np.array(episode["roles"][: z.shape[0]])
        trace = z[:, slot, args.pair].astype(np.float32)
        values = []
        for turn in episode["turns"]:
            if "start" not in turn:
                continue
            low, high = turn["start"], min(turn["end"], z.shape[0])
            if high <= low:
                continue
            window = roles[low:high]
            keep = window == "thinking"
            if not keep.any():
                keep = np.isin(window, GENERATED)
            if not keep.any():
                continue
            values.append(float(trace[low:high][keep].mean()))
        if not values:
            continue
        counts[ending] += 1
        for index, value in enumerate(values):
            forward[ending][index].append(value)
        for index, value in enumerate(reversed(values)):
            backward[ending][-1 - index].append(value)

    def series(store: dict[int, list[float]], keys: list[int]) -> dict:
        return {
            "x": keys,
            "mean": [float(np.mean(store[k])) for k in keys],
            "sem": [
                float(np.std(store[k], ddof=1) / np.sqrt(len(store[k]))) if len(store[k]) > 1 else 0.0 for k in keys
            ],
            "n": [len(store[k]) for k in keys],
        }

    result = {
        "pair": args.pair,
        "concept": name,
        "class": klass,
        "layer": args.layer,
        "episodes": dict(counts),
        "forward": {},
        "backward": {},
    }
    for group in GROUPS:
        ahead = [k for k in sorted(forward[group]) if k < args.max_turns and len(forward[group][k]) >= 5]
        behind = [k for k in sorted(backward[group]) if k >= -args.max_turns and len(backward[group][k]) >= 5]
        result["forward"][group] = series(forward[group], ahead)
        result["backward"][group] = series(backward[group], behind)

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"trace-{args.pair}-L{args.layer}.json"
    target.write_text(json.dumps(result, indent=1))
    log.info(f"wrote {target}  episodes {dict(counts)}")

    for group, label in GROUPS.items():
        b = result["backward"][group]
        if b["x"]:
            spots = ", ".join(f"{x}:{m:+.3f}" for x, m in list(zip(b["x"], b["mean"]))[-6:])
            log.info(f"  {label:<14} last turns  {spots}")


if __name__ == "__main__":
    main()
