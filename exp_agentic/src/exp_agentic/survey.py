#! /usr/bin/env python


import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from exp_agentic.readout import find

log = logging.getLogger("survey")

LAYERS = (18, 25)
THRESHOLD = 4.5
BOUNDARY = 8
GENERATED = ("thinking", "tool_call", "answer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    rows = pq.read_table(find("pairs.parquet")).to_pylist()
    names = [f"{row['concept']} || {row['antagonist']}" for row in rows]
    classes = [row["class_name"] for row in rows]

    episodes = sorted(args.dir.glob("*.z.npy"))
    log.info(f"{len(episodes)} scored episodes in {args.dir}")

    per_concept = {
        layer: {
            "peak": np.zeros((len(episodes), len(names)), dtype=np.float32),
            "atboundary": np.zeros((len(episodes), len(names)), dtype=bool),
        }
        for layer in LAYERS
    }
    labels = []

    for index, path in enumerate(episodes):
        record = Path(str(path).replace(".z.npy", ".json"))
        episode = json.loads(record.read_text())
        starts = np.array([turn["start"] for turn in episode["turns"] if "start" in turn] or [0])
        z = np.abs(np.load(path).astype(np.float32))

        roles = np.array(episode["roles"][: z.shape[0]])
        mine = np.isin(roles, GENERATED)
        if not mine.any():
            continue
        positions = np.flatnonzero(mine)
        z = z[mine]

        for slot, layer in enumerate(LAYERS):
            plane = z[:, slot, :]
            where = positions[plane.argmax(axis=0)]
            per_concept[layer]["peak"][index] = plane.max(axis=0)
            gap = np.abs(where[:, None] - starts[None, :]).min(axis=1)
            per_concept[layer]["atboundary"][index] = gap <= BOUNDARY

        labels.append(
            {
                "episode": record.stem,
                "ending": episode.get("ending"),
                "hacked": episode.get("ending") == "submit",
                "tokens": len(episode.get("ids", [])),
            }
        )
        if (index + 1) % 50 == 0:
            log.info(f"{index + 1}/{len(episodes)}")

    hacked = np.array([entry["hacked"] for entry in labels])
    out = {"episodes": len(episodes), "hacked": int(hacked.sum()), "threshold": THRESHOLD, "concepts": []}

    for pair in range(len(names)):
        entry = {"pair": pair, "concept": names[pair], "class": classes[pair], "layers": {}}
        for layer in LAYERS:
            peak = per_concept[layer]["peak"][:, pair]
            boundary = per_concept[layer]["atboundary"][:, pair]
            entry["layers"][f"L{layer}"] = {
                "mean_peak": round(float(peak.mean()), 3),
                "max_peak": round(float(peak.max()), 3),
                "episodes_over_threshold": int((peak > THRESHOLD).sum()),
                "fraction_at_turn_boundary": round(float(boundary.mean()), 3),
                "mean_peak_hacked": round(float(peak[hacked].mean()), 3) if hacked.any() else None,
                "mean_peak_other": round(float(peak[~hacked].mean()), 3) if (~hacked).any() else None,
            }
        entry["gap_hacked_minus_other"] = (
            round(float(entry["layers"]["L25"]["mean_peak_hacked"] - entry["layers"]["L25"]["mean_peak_other"]), 3)
            if hacked.any() and (~hacked).any()
            else None
        )
        out["concepts"].append(entry)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    log.info(f"wrote {args.out}")

    ranked = sorted(out["concepts"], key=lambda e: -e["layers"]["L25"]["mean_peak"])
    log.info("top 15 by L25 mean peak |z| (boundary fraction in brackets):")
    for entry in ranked[:15]:
        info = entry["layers"]["L25"]
        log.info(f"  {info['mean_peak']:5.2f} [{info['fraction_at_turn_boundary']:.2f}]  {entry['concept'][:58]}")


if __name__ == "__main__":
    main()
