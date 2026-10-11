#! /usr/bin/env python
import argparse
import json
from pathlib import Path

ALPHAS = (0.10, -0.10)

ARMS = [
    {
        "name": "trained_thinking",
        "direction": "file:vectors/trained-L18-r50.npy",
        "why": "true DPO objective, deliberation window; reproducible across hyperparameters (cos .925)",
    },
    {
        "name": "trained_all",
        "direction": "file:vectors/trained-L18-r15.npy",
        "why": "true DPO, all emitted tokens; scores higher but reads the tool call too",
    },
    {
        "name": "grpo_hack_giveup",
        "direction": "file:vectors/grpo-L18-hack_vs_giveup-mean.npy",
        "why": "one-step GRPO; drove the hack rate to 0.944 in the first sweep",
    },
    {
        "name": "dpo_linearized",
        "direction": "file:vectors/vector-L18-mean.npy",
        "why": "linearized DPO, the first sweep's other mover",
    },
    {
        "name": "story_grad_872",
        "direction": "file:vectors/gradvec-872-L18.npy",
        "why": "gradient-space 'shortcut acceptance || rejection'",
    },
    {
        "name": "story_pub_872",
        "direction": "file:vectors/pub872-L18.npy",
        "why": "PUBLISHED activation-space vector, identical concept",
    },
    {
        "name": "ctl_random0",
        "direction": "file:vectors/ctl-random0-L18.npy",
        "why": "matched-norm random direction: steering ANY direction moves generation length",
    },
    {
        "name": "ctl_random1",
        "direction": "file:vectors/ctl-random1-L18.npy",
        "why": "a second random draw, so the control has its own spread",
    },
    {
        "name": "ctl_shared",
        "direction": "file:vectors/ctl-shared-L18.npy",
        "why": "normalized mean of all 1036 published directions",
    },
]

ABLATE = ["trained_thinking", "grpo_hack_giveup", "story_grad_872", "ctl_random0"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("arms.json"))
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--seed-from", type=int, default=1000)
    args = parser.parse_args()

    jobs = []
    for alpha in ALPHAS:
        for arm in ARMS:
            jobs.append(
                {
                    "name": arm["name"],
                    "direction": arm["direction"],
                    "mode": "add",
                    "alpha": alpha,
                    "priority": 0 if alpha > 0 else 1,
                }
            )
    for arm in ARMS:
        if arm["name"] in ABLATE:
            jobs.append(
                {"name": arm["name"], "direction": arm["direction"], "mode": "project", "alpha": 0.0, "priority": 2}
            )

    plan = {
        "seeds": args.seeds,
        "seed_from": args.seed_from,
        "arms": ARMS,
        "jobs": jobs,
        "episodes": len(jobs) * args.seeds,
    }
    args.out.write_text(json.dumps(plan, indent=1))
    print(f"{len(jobs)} arm-configurations x {args.seeds} seeds = {plan['episodes']} episodes")
    for job in jobs:
        print(f"  {job['name']:<20} {job['mode']:<8} alpha={job['alpha']:+.2f}  {job['direction']}")


if __name__ == "__main__":
    main()
