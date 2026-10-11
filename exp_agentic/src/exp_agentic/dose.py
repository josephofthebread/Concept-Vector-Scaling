#! /usr/bin/env python
import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("dose")

ENDINGS = ("submit", "give_up", "degenerate", "max_turns", "context_exhausted", "parse_failures")
HACK = "submit"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True, help="directory of evaluated episodes")
    parser.add_argument("--out", type=Path, default=Path("analysis/dose.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cells: dict[tuple[str, str, float | None], list[dict]] = defaultdict(list)
    for path in sorted(args.dir.glob("*.json")):
        episode = json.loads(path.read_text())
        if "ending" not in episode:
            continue
        steering = episode.get("steering") or {}
        arm = steering.get("pair") or "none"
        if isinstance(arm, str) and arm.startswith("file:"):
            arm = Path(arm.removeprefix("file:")).stem
        cells[(arm, steering.get("mode", "add"), steering.get("alpha"))].append(episode)

    rows = []
    for (arm, mode, alpha), episodes in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0)):
        counts = {e: 0 for e in ENDINGS}
        for episode in episodes:
            counts[episode["ending"]] = counts.get(episode["ending"], 0) + 1
        total = len(episodes)
        if not total:
            low, high = (0.0, 0.0)
        else:
            p = counts[HACK] / total
            denominator = 1 + 1.96 * 1.96 / total
            center = (p + 1.96 * 1.96 / (2 * total)) / denominator
            spread = 1.96 * math.sqrt(p * (1 - p) / total + 1.96 * 1.96 / (4 * total * total)) / denominator
            low, high = (max(0.0, center - spread), min(1.0, center + spread))
        rows.append(
            {
                "arm": arm,
                "mode": mode,
                "alpha": alpha,
                "n": total,
                "hack": counts[HACK] / total if total else 0.0,
                "hack_lo": low,
                "hack_hi": high,
                "give_up": counts["give_up"] / total if total else 0.0,
                "degenerate": counts["degenerate"] / total if total else 0.0,
                "turns": sum(len(e.get("turns", [])) for e in episodes) / total if total else 0.0,
                "counts": counts,
            }
        )

    log.info(f"{'arm':<22}{'mode':<9}{'alpha':>7}{'n':>5}{'hack':>18}{'give_up':>9}{'degenerate':>11}{'turns':>7}")
    for row in rows:
        alpha = "-" if row["alpha"] is None else f"{row['alpha']:+.2f}"
        band = f"{row['hack']:.3f} [{row['hack_lo']:.2f},{row['hack_hi']:.2f}]"
        log.info(
            f"{row['arm']:<22}{row['mode']:<9}{alpha:>7}{row['n']:>5}{band:>18}"
            f"{row['give_up']:>9.3f}{row['degenerate']:>11.3f}{row['turns']:>7.1f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cells": rows}, indent=1))
    log.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
