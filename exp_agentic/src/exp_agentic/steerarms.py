#! /usr/bin/env python


import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector", required=True, help="path inside the job, e.g. vectors/multi-L18-mean.npy")
    parser.add_argument("--out", type=Path, default=Path("arms.json"))
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--seed-from", type=int, default=9000)
    args = parser.parse_args()

    candidate = f"file:{args.vector}"
    control = "file:vectors/ctl-random0-L18.npy"
    cells = [
        ("held_out_plus", "12_shortcuts", "sumsquares", candidate, 0.10),
        ("held_out_ctl", "12_shortcuts", "sumsquares", control, 0.10),
        ("in_domain_plus", "12_shortcuts", "matdet", candidate, 0.10),
        ("in_domain_ctl", "12_shortcuts", "matdet", control, 0.10),
        ("in_domain_minus", "12_shortcuts", "matdet", candidate, -0.10),
    ]
    jobs = [
        {"name": name, "workload": workload, "variant": variant, "direction": direction, "mode": "add", "alpha": alpha}
        for name, workload, variant, direction, alpha in cells
    ]
    plan = {"seeds": args.seeds, "seed_from": args.seed_from, "jobs": jobs, "episodes": len(jobs) * args.seeds}
    args.out.write_text(json.dumps(plan, indent=1))
    print(f"{len(jobs)} cells x {args.seeds} seeds = {plan['episodes']} episodes")
    for job in jobs:
        print(f"  {job['name']:<18} {job['variant']:<12} alpha={job['alpha']:+.2f}  {job['direction']}")
    print("\nbaselines from the screen, not re-run: matdet 0.357 (n=14), sumsquares 0.071 (n=14)")


if __name__ == "__main__":
    main()
