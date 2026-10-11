#! /usr/bin/env python
import argparse
import importlib
import json
import logging
import time
import zlib
from pathlib import Path

import torch

from exp_agentic.agent import execute, run_episode
from exp_agentic.model import load
from exp_agentic.sandbox import Sandbox

log = logging.getLogger("fork")

MUTATING = ("write_file",)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("episodes/gate"))
    parser.add_argument("--out", type=Path, default=Path("episodes/forked"))
    parser.add_argument("--workload", default="01_impossible_tests")
    parser.add_argument("--variant", default="gate")
    parser.add_argument("--k", type=int, default=8, help="continuations per branch point")
    parser.add_argument("--back", type=int, default=3, help="turns before the ending to branch at")
    parser.add_argument("--floor", type=int, default=2, help="earliest turn to branch at")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--seed-base", type=int, default=50000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    workload = importlib.import_module(f"exp_agentic.workloads.{args.workload}")
    model, tokenizer = load(device=args.device, dtype=args.dtype)
    args.out.mkdir(parents=True, exist_ok=True)

    priority = {"submit": 0, "give_up": 1, "degenerate": 2}
    catalog = []
    for path in sorted(args.dir.glob("*.json")):
        if path.name.startswith("._"):
            continue
        catalog.append((priority.get(json.loads(path.read_text()).get("ending"), 9), path.name, path))
    ordered = [path for _, _, path in sorted(catalog)]
    sources = [p for i, p in enumerate(ordered) if i % args.shards == args.shard]
    log.info(
        f"shard {args.shard}/{args.shards}: {len(sources)} source trajectories, k={args.k}, "
        f"first {sources[0].stem if sources else '-'}"
    )

    made = 0
    for index, path in enumerate(sources, start=1):
        episode = json.loads(path.read_text())
        turns = [t for t in episode.get("turns", []) if "start" in t]
        branch = max(args.floor, len(turns) - args.back)
        if branch >= len(turns):
            log.info(f"{path.stem}: only {len(turns)} turns, no room to branch")
            continue

        for slot in range(args.k):
            stem = f"{path.stem}-b{branch}-k{slot}"
            target = args.out / f"{stem}.json"
            if target.exists():
                continue
            root = args.out / stem
            started = time.monotonic()
            try:
                root.mkdir(parents=True, exist_ok=True)
                overlay = workload.VARIANTS[episode.get("variant", "gate")]
                for relative, content in {**workload.FILES, **overlay["files"]}.items():
                    destination = root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(content)

                prefix = episode["turns"][:branch]
                distinct: set[str] = set()
                stale, given_up = 0, False
                with Sandbox(root) as box:
                    for record in prefix:
                        name, arguments = record.get("tool"), record.get("arguments") or {}
                        if name in MUTATING:
                            execute(name, arguments, root, box, workload)
                            body = " ".join(arguments.get("content", "").split())
                            stale = 0 if body not in distinct else stale + 1
                            distinct.add(body)
                        else:
                            stale += 1
                        if record.get("event") == "give_up_refused":
                            given_up = True

                cut = episode["turns"][branch]["start"]
                state = {
                    "ids": episode["ids"][:cut],
                    "roles": episode["roles"][:cut],
                    "turns": prefix,
                    "distinct": distinct,
                    "given_up": given_up,
                    "stale": stale,
                    "turn": branch,
                }
                seed = args.seed_base + zlib.crc32(stem.encode()) % 100000
                torch.manual_seed(seed)
                produced = run_episode(
                    model, tokenizer, workload, root, seed, args.variant, None, args.max_turns, resume=state
                )
            except Exception as problem:
                log.warning(f"{stem}: {type(problem).__name__}: {problem}")
                continue

            produced["variant"] = args.variant
            produced["fork"] = {
                "source": path.stem,
                "source_ending": episode.get("ending"),
                "branch_turn": branch,
                "slot": slot,
                "prefix_tokens": len(state["ids"]),
            }
            target.write_text(json.dumps(produced))
            made += 1
            log.info(
                f"{stem}: {produced['ending']} in {len(produced['turns']) - branch} new turns, "
                f"{time.monotonic() - started:.0f}s"
            )

        if index % 5 == 0:
            log.info(f"--- {index}/{len(sources)} sources, {made} continuations written ---")

    log.info(f"shard {args.shard} complete: {made} continuations under {args.out}")


if __name__ == "__main__":
    main()
