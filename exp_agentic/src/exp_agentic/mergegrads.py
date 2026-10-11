#! /usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True, help="directory of per-shard grads-*.npz")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--legacy",
        action="append",
        default=[],
        metavar="PATH=WORKLOAD",
        help="an existing archive to pool in under a given workload name",
    )
    parser.add_argument(
        "--relabel",
        type=Path,
        default=None,
        help="shortcutprobe.py --out json; its outcomes override the ending-derived class",
    )
    parser.add_argument(
        "--honest-includes-degenerate",
        action="append",
        default=[],
        metavar="WORKLOAD",
        help="treat degenerate as the non-hack class here",
    )
    args = parser.parse_args()

    verdicts: dict[str, dict] = {}
    if args.relabel:
        payload = json.loads(args.relabel.read_text())
        for name, verdict in payload.get("episodes", payload).items():
            verdicts[Path(name).stem] = verdict

    blocks, meta, layer = [], [], None
    for entry in args.legacy:
        path_text, _, label = entry.partition("=")
        with np.load(Path(path_text), allow_pickle=True) as archive:
            rows = archive["g"]
            found = int(archive["layer"])
            if layer is None:
                layer = found
            elif found != layer:
                raise SystemExit(f"{path_text} is layer {found}, expected {layer} -- refusing to pool")
            legacy_meta = [dict(m) for m in archive["meta"]]
        for row in legacy_meta:
            row.setdefault("workload", label or "legacy")
        blocks.append(rows)
        meta.extend(legacy_meta)
        print(f"{Path(path_text).name:<28} {rows.shape[0]:>4} trajectories  (labelled {label!r})")
    for path in sorted(args.dir.glob("*.npz")):
        with np.load(path, allow_pickle=True) as archive:
            rows = archive["g"]
            if rows.shape[0] == 0:
                print(f"{path.name:<28} empty, skipped")
                continue
            found = int(archive["layer"])
            if layer is None:
                layer = found
            elif found != layer:
                raise SystemExit(f"{path.name} is layer {found}, expected {layer} -- refusing to pool")
            blocks.append(rows)
            meta.extend(list(archive["meta"]))
            print(f"{path.name:<28} {rows.shape[0]:>4} trajectories")

    if verdicts:
        changed = Counter()
        for row in meta:
            verdict = verdicts.get(row.get("stem", ""))
            if not verdict or verdict.get("outcome") is None:
                continue
            was = row.get("group")
            if verdict["outcome"] >= 2:
                now = "hack"
            else:
                now = "giveup" if row.get("ending") == "give_up" else "degenerate"
            if now != was:
                changed[f"{was} -> {now}"] += 1
                row["group"] = now
        print(f"relabelled from probe verdicts: {dict(changed) or 'nothing changed'}")

    if args.honest_includes_degenerate:
        collapsed = Counter()
        for row in meta:
            if row.get("workload") in args.honest_includes_degenerate and row.get("group") == "degenerate":
                row["group"] = "giveup"
                collapsed[row["workload"]] += 1
        print(f"degenerate folded into the honest class: {dict(collapsed) or 'nothing'}")

    if not blocks:
        raise SystemExit("no non-empty shards found")
    stacked = np.concatenate(blocks, axis=0)
    if stacked.shape[0] != len(meta):
        raise SystemExit(f"{stacked.shape[0]} rows against {len(meta)} metadata entries")

    np.savez_compressed(args.out, g=stacked, meta=np.array(meta, dtype=object), layer=np.array(layer))
    workloads = Counter(m.get("workload", "unknown") for m in meta)
    groups = Counter(m.get("group") for m in meta)
    prefixes = {m["prefix"] for m in meta}
    by_prefix: dict[str, set] = {}
    for row in meta:
        by_prefix.setdefault(row["prefix"], set()).add(row.get("group"))
    mixed = [p for p, kinds in by_prefix.items() if {"hack", "giveup"} <= kinds]
    mixed_by_workload = Counter(next(m["workload"] for m in meta if m["prefix"] == p) for p in mixed)

    print(f"\n{stacked.shape[0]} trajectories, layer {layer}")
    print(f"groups: {dict(groups)}")
    print(f"workloads: {dict(workloads)}")
    print(f"{len(prefixes)} branch points, {len(mixed)} MIXED (both a hack and a give-up)")
    print(f"mixed by workload: {dict(mixed_by_workload)}")
    print(
        json.dumps(
            {
                "trajectories": int(stacked.shape[0]),
                "branch_points": len(prefixes),
                "mixed": len(mixed),
                "mixed_by_workload": dict(mixed_by_workload),
            }
        )
    )


if __name__ == "__main__":
    main()
