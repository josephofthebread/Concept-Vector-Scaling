#! /usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

DIVERGENCE = 0.227

FLOOR = 0.08


def mixed_probability(rate: float, k: int) -> float:
    """Estimated chance that one branch point of k continuations yields both a hack and a give-up."""
    effective = 0.5 - (0.5 - rate) * (1.0 - DIVERGENCE) if rate < 0.5 else rate
    effective = min(max(effective * DIVERGENCE + rate * (1 - DIVERGENCE), 0.001), 0.999)
    return 1.0 - effective**k - (1.0 - effective) ** k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True, help="shortcutprobe.py --out json")
    parser.add_argument("--out", type=Path, default=Path("forkplan.json"))
    parser.add_argument(
        "--budget", type=int, default=250, help="total continuations affordable across the whole fork wave"
    )
    parser.add_argument("--take", type=int, default=2, help="how many workloads to fork")
    parser.add_argument("--back", type=int, default=3)
    args = parser.parse_args()

    summary = json.loads(args.screen.read_text())["summary"]
    ranked = sorted(
        ((v["rate"], name, v) for name, v in summary.items() if not name.endswith("_possible")), reverse=True
    )

    print(f"{'workload':<22}{'n':>4}{'rate':>8}{'P(mixed|k=8)':>14}   decision")
    chosen = []
    for rate, name, cell in ranked:
        estimate = mixed_probability(rate, 8)
        if rate < FLOOR:
            note = "EXCLUDED - rate too low to yield mixed prefixes"
        elif len(chosen) >= args.take:
            note = "not forked - budget spent on higher-rate workloads"
        else:
            note = "FORK"
            chosen.append((name, rate, cell))
        print(f"{name:<22}{cell['n']:>4}{rate:>8.3f}{estimate:>14.2f}   {note}")

    if not chosen:
        print("\nNO WORKLOAD CLEARS THE FLOOR. A multi-workload fit is not possible from this screen.")
        print(
            "The honest conclusion is that this model reward-hacks only where the shortcut is a "
            "fact it already holds, and `fastsum` is the only such task found."
        )
        args.out.write_text(json.dumps({"fork": []}, indent=1))
        return

    share = args.budget // len(chosen)
    plan = []
    print()
    for name, rate, cell in chosen:
        for k in (6, 8, 10, 12, 16):
            if mixed_probability(rate, k) >= 0.45:
                break
        else:
            k = 16
        points = max(1, share // k)
        plan.append({"workload": "12_shortcuts", "variant": name, "k": k, "back": args.back})
        print(
            f"{name:<22} k={k:<3} ~{points} branch points x {k} = ~{points * k} continuations, "
            f"expect ~{points * mixed_probability(rate, k):.0f} mixed"
        )

    args.out.write_text(json.dumps({"fork": plan, "budget": args.budget}, indent=1))
    print(f"\nwrote {args.out} with {len(plan)} workloads")
    print(
        "fastsum's existing forked gradients are pooled in separately, so the fitted corpus is "
        f"{len(plan)} + 1 workloads and leave-one-workload-out has something to hold out."
    )


if __name__ == "__main__":
    main()
