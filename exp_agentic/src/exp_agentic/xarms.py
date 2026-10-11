#! /usr/bin/env python


import argparse
import json
from pathlib import Path

GRPO = "file:vectors/grpo-L18-hack_vs_giveup-mean.npy"
DPO = "file:vectors/vector-L18-mean.npy"
RANDOM = "file:vectors/ctl-random0-L18.npy"

CELLS = [
    ("grpo_plus", "gate", GRPO, 0.10, "the transfer claim: the confirmed vector, same layer, same alpha, new content"),
    (
        "baseline",
        "gate",
        None,
        0.0,
        "fastsort's own unsteered hack rate -- the floor every other gate cell is read against",
    ),
    (
        "ctl_random0_plus",
        "gate",
        RANDOM,
        0.10,
        "matched-norm random direction: any perturbation of this size shortens and simplifies output",
    ),
    (
        "possible_grpo_plus",
        "possible",
        GRPO,
        0.10,
        "specificity: does it INDUCE hacking where an honest path is available and works?",
    ),
    (
        "possible_baseline",
        "possible",
        None,
        0.0,
        "the solvable task's own floor; expected near zero, which is what makes the cell above readable",
    ),
    (
        "possible_ctl_random0_plus",
        "possible",
        RANDOM,
        0.10,
        "the same control on the solvable task, so gratuitous hacking has a noise floor too",
    ),
    (
        "grpo_minus",
        "gate",
        GRPO,
        -0.10,
        "sign test: on fastsum this drove the rate to 0.000, and a one-sided effect is weak evidence",
    ),
    (
        "ctl_random0_minus",
        "gate",
        RANDOM,
        -0.10,
        "the negative control, absent from every sweep so far -- generic collapse has never been measured",
    ),
    (
        "dpo_linearized_minus",
        "gate",
        DPO,
        -0.10,
        "the second estimator of the same axis (cos +0.532); its preferred branch is give_up, so it "
        "steers TOWARD hacking at negative alpha",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("arms.json"))
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--seed-from", type=int, default=5000)
    args = parser.parse_args()

    jobs = [
        {
            "name": name,
            "workload": "11_fastsort",
            "variant": variant,
            "direction": direction,
            "mode": "add",
            "alpha": alpha,
            "why": why,
        }
        for name, variant, direction, alpha, why in CELLS
    ]
    plan = {"seeds": args.seeds, "seed_from": args.seed_from, "jobs": jobs, "episodes": len(jobs) * args.seeds}
    args.out.write_text(json.dumps(plan, indent=1))

    print(f"{len(jobs)} cells x {args.seeds} seeds = {plan['episodes']} episodes")
    for job in jobs:
        direction = job["direction"] or "(unsteered)"
        print(f"  {job['name']:<26} {job['variant']:<9} alpha={job['alpha']:+.2f}  {direction}")


if __name__ == "__main__":
    main()
