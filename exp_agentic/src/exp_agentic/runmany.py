from __future__ import annotations

import argparse
import importlib
import json
import logging
import time
from pathlib import Path

from exp_agentic.agent import run_episode
from exp_agentic.model import load

CONFIG = {
    "A": ("on", "refuse-once"),
    "B": ("on", "accept"),
    "C": ("on", "none"),
    "D": ("off", "refuse-once"),
}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("runmany")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIG))
    ap.add_argument("--workload", default="01_impossible_tests")
    ap.add_argument("--variant", default="gate")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", type=Path, default=Path("episodes4"))
    args = ap.parse_args()

    gate_override, giveup_mode = CONFIG[args.config]
    workload = importlib.import_module(f"exp_agentic.workloads.{args.workload}")
    model, tokenizer = load(device=args.device, dtype=args.dtype)
    args.out.mkdir(parents=True, exist_ok=True)
    log.info(
        "config %s (gate=%s giveup=%s) x %d seeds -> %s", args.config, gate_override, giveup_mode, args.seeds, args.out
    )

    for seed in range(args.seeds):
        name = f"{workload.NAME}-{args.variant}-{args.config}-seed{seed}"
        record = args.out / f"{name}.json"
        if record.exists():
            log.info("skip %s (exists)", name)
            continue
        root = args.out / name
        t0 = time.monotonic()
        try:
            episode = run_episode(
                model,
                tokenizer,
                workload,
                root,
                seed,
                args.variant,
                None,
                40,
                gate_override=gate_override,
                giveup_mode=giveup_mode,
            )
        except Exception:
            import traceback

            log.error("seed %d failed:\n%s", seed, traceback.format_exc())
            continue
        episode["variant"] = args.variant
        record.write_text(json.dumps(episode, indent=2))
        log.info(
            "seed %d: %s, %d turns, %d tok (%.0fs) -> %s",
            seed,
            episode["ending"],
            len(episode["turns"]),
            len(episode["ids"]),
            time.monotonic() - t0,
            record.name,
        )
    log.info("config %s done", args.config)


if __name__ == "__main__":
    main()
