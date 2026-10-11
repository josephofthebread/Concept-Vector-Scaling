#! /usr/bin/env python
import argparse
import glob
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from exp_agentic.readout import find

log = logging.getLogger("peek")
LAYERS = {18: 0, 25: 1}
GENERATED = ("thinking", "tool_call", "answer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("episodes/gate"))
    parser.add_argument("--pair", type=int, help="concept id to inspect")
    parser.add_argument("--search", help="find pair ids whose name contains this text")
    parser.add_argument("--layer", type=int, default=25, choices=[18, 25])
    parser.add_argument("--top", type=int, default=5, help="how many peak episodes to show")
    parser.add_argument("--span", type=int, default=40, help="tokens either side of the peak")
    parser.add_argument("--null", action="store_true", help="report the noise floor for these lengths")
    parser.add_argument(
        "--all-tokens",
        action="store_true",
        help="include the prompt and tool results; off by default because the prompt is identical in every episode",
    )
    parser.add_argument(
        "--only",
        choices=["hacked", "give_up", "degenerate"],
        help="restrict to an outcome; the stored labels are submit/give_up/degenerate",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    rows = pq.read_table(find("pairs.parquet")).to_pylist()
    entries = [
        {"pair": index, "concept": f"{row['concept']} || {row['antagonist']}", "class": row["class_name"]}
        for index, row in enumerate(rows)
    ]

    if args.search:
        needle = args.search.lower()
        for entry in entries:
            if needle in entry["concept"].lower() or needle in entry["class"].lower():
                print(f"{entry['pair']:5d}  {entry['concept']}   [{entry['class']}]")
        return

    if args.pair is None:
        raise SystemExit("give --pair or --search")

    slot = LAYERS[args.layer]
    scored = sorted(glob.glob(str(args.dir / "*.z.npy")))
    found = []
    nulls = []
    for path in scored:
        record = Path(path.replace(".z.npy", ".json"))
        episode = json.loads(record.read_text())
        ending = episode.get("ending")
        if args.only == "hacked" and ending != "submit":
            continue
        if args.only and args.only != "hacked" and ending != args.only:
            continue
        loaded = np.load(path)
        roles = np.array(episode["roles"][: loaded.shape[0]])
        keep = np.arange(loaded.shape[0]) if args.all_tokens else np.flatnonzero(np.isin(roles, GENERATED))
        if not len(keep):
            continue
        trace = np.abs(loaded[keep, slot, args.pair].astype(np.float32))
        found.append((float(trace.max()), int(keep[trace.argmax()]), record, ending))
        if args.null:
            plane = np.abs(loaded[keep][:, slot, :].astype(np.float32))
            nulls.append(float(np.median(plane.max(axis=0))))

    if not found:
        raise SystemExit(f"no episodes matched --only {args.only!r} in {args.dir}")
    found.sort(reverse=True)
    entry = entries[args.pair]
    print(f"pair {args.pair}: {entry['concept']}   [{entry['class']}]")
    print(f"L{args.layer}, {len(found)} episodes" + (f", filtered to {args.only}" if args.only else ""))
    if nulls:
        print(f"NOISE FLOOR: median concept in these episodes peaks at |z| = {np.median(nulls):.2f}")
    peaks = np.array([value for value, _, _, _ in found])
    print(f"this concept: mean peak {peaks.mean():.2f}, max {peaks.max():.2f}")
    print()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    for value, position, record, ending in found[: args.top]:
        episode = json.loads(record.read_text())
        ids = episode["ids"]
        low, high = max(0, position - args.span), min(len(ids), position + args.span)
        before = tokenizer.decode(ids[low:position], skip_special_tokens=False)
        at = tokenizer.decode(ids[position : position + 1], skip_special_tokens=False)
        after = tokenizer.decode(ids[position + 1 : high], skip_special_tokens=False)

        location = "opening prompt"
        for turn in episode.get("turns", []):
            if "start" in turn and turn["start"] <= position < turn["end"]:
                location = f"turn {turn['turn']} ({turn.get('event')} {turn.get('tool') or ''})".strip()
                break
        else:
            if episode.get("turns") and "start" in episode["turns"][0] and position >= episode["turns"][0]["start"]:
                location = "tool result"
        role = episode["roles"][position] if position < len(episode.get("roles", [])) else "?"
        where, text = f"{location}, role={role}, token {position}/{len(ids)}", f"{before}>>>{at}<<<{after}"
        print("=" * 78)
        print(f"|z| = {value:.2f}   {record.stem}   ending={ending}")
        print(f"   {where}")
        print("-" * 78)
        print(text)
        print()


if __name__ == "__main__":
    main()
