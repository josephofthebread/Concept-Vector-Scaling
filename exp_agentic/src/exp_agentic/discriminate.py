#! /usr/bin/env python
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from exp_agentic.rank import SPOKEN, zed
from exp_agentic.readout import LAYERS, find

log = logging.getLogger("discriminate")

HACKED = {
    "impossible_tests-judge-seed3",
    "impossible_tests-judge-seed7",
    "impossible_tests-readme-seed6",
    "impossible_tests-readme-seed7",
}
DEAD: set[str] = set()

WATCH = [
    "shortcut acceptance || shortcut rejection",
    "honest admission of not knowing || forced solution under pressure",
    "transparency about limits || bluffing about capabilities",
    "appeasement || standing firm",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("episodes/stage1"))
    parser.add_argument("--layer", type=int, default=25, choices=sorted(LAYERS))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    slot = list(LAYERS).index(args.layer)

    rows = pq.read_table(find("pairs.parquet")).to_pylist()
    names = [f"{row['concept']} || {row['antagonist']}" for row in rows]
    lookup = {name: index for index, name in enumerate(names)}
    watched = [(name, lookup[name]) for name in WATCH if name in lookup]
    for name in WATCH:
        if name not in lookup:
            log.info(f"not found in pairs.parquet, skipping: {name}")

    groups: dict[str, dict[str, list[float]]] = {name: {"hacked": [], "honest": []} for name, _ in watched}
    hack_positions = []

    for record in sorted(args.dir.glob("*.json")):
        blob = record.with_suffix(".scores.npy")
        if not blob.is_file() or record.stem in DEAD:
            continue
        episode = json.loads(record.read_text())
        roles = np.array(episode["roles"])
        spoken = np.isin(roles, SPOKEN)
        if not spoken.any():
            continue
        standard = zed(np.load(blob))[:, slot, :]
        label = "hacked" if record.stem in HACKED else "honest"
        for name, pair in watched:
            groups[name][label].append(float(np.abs(standard[spoken, pair]).max()))

        if label == "hacked":
            pair = lookup[WATCH[0]]
            column = np.where(spoken, np.abs(standard[:, pair]), -np.inf)
            hack_positions.append((record.stem, int(np.argmax(column)), len(roles), roles[int(np.argmax(column))]))

    sample = next(iter(groups.values()))
    hacked_label = f"hacked (n={len(sample['hacked'])})"
    honest_label = f"honest (n={len(sample['honest'])})"
    print(f"\nL{args.layer}: peak |z| on model-produced tokens\n")
    print(f"{'concept':<58} {hacked_label:<22} {honest_label:<22} overlap?")
    for name, _ in watched:
        hacked, honest = groups[name]["hacked"], groups[name]["honest"]
        if not hacked or not honest:
            continue
        separated = min(hacked) > max(honest) or max(hacked) < min(honest)
        print(
            f"{name[:56]:<58} {np.mean(hacked):.2f} [{min(hacked):.2f}-{max(hacked):.2f}]   ".ljust(22)
            + f"{np.mean(honest):.2f} [{min(honest):.2f}-{max(honest):.2f}]   ".ljust(22)
            + ("SEPARATED" if separated else "overlaps")
        )

    print("\nwhere the shortcut concept peaks in the hacking episodes:")
    for stem, at, total, role in hack_positions:
        print(f"  {stem}: token {at} of {total} ({at / total:.0%} through), role={role}")


if __name__ == "__main__":
    main()
