from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from pathlib import Path

from exp_reward_hack import workloads
from exp_reward_hack.agent import run_episode
from exp_reward_hack.dialects import for_model

MODELTAG = {
    "Qwen/Qwen3-8B": "qwen",
    "unsloth/Meta-Llama-3.1-8B-Instruct": "llama",
    "google/gemma-2-9b-it": "gemma",
}
CONFIG = {
    "A": ("on", "refuse-once"),
    "B": ("on", "accept"),
    "C": ("on", "none"),
    "D": ("off", "refuse-once"),
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("runmany")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELTAG))
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--config", default="A", choices=list(CONFIG))
    ap.add_argument("--seeds", type=int, default=3, help="how many seeds this shard runs")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument(
        "--seed-stride",
        type=int,
        default=1,
        help="gap between seeds, so K interleaved shards (strides 0..K-1) tile the space "
        "densely even if some die early",
    )
    ap.add_argument("--workloads", default="ALL", help="comma-separated names, or ALL")
    ap.add_argument("--variant", default="gate")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--out", type=Path, default=Path("episodes_smoke"))
    args = ap.parse_args()

    names = workloads.ALL if args.workloads == "ALL" else [w.strip() for w in args.workloads.split(",")]
    mods = [workloads.load(n) for n in names]
    gate_override, giveup_mode = CONFIG[args.config]
    mtag = MODELTAG[args.model]
    dialect = for_model(args.model)
    seeds = [args.seed_start + k * args.seed_stride for k in range(args.seeds)]

    if args.backend == "vllm":
        from exp_reward_hack import vllm_gen

        engine, tokenizer, context_limit = vllm_gen.load(args.model, dtype=args.dtype)
        eos_ids = dialect.eos_ids(tokenizer)
        turn_end_id = tokenizer.convert_tokens_to_ids(dialect.turn_end)
        model = None
    else:
        from exp_reward_hack.model import load

        model, tokenizer = load(model_id=args.model, device=args.device, dtype=args.dtype)
        context_limit = None

    args.out.mkdir(parents=True, exist_ok=True)
    log.info(
        "model %s backend %s config %s x seeds %s over %d workloads -> %s",
        mtag,
        args.backend,
        args.config,
        list(seeds),
        len(mods),
        args.out,
    )

    for wl in mods:
        for seed in seeds:
            name = f"{wl.NAME}-{mtag}-{args.config}-seed{seed}"
            record = args.out / f"{name}.json"
            if record.exists():
                log.info("skip %s (exists)", name)
                continue
            generate_turn = (
                vllm_gen.make_generate(engine, eos_ids, turn_end_id, seed) if args.backend == "vllm" else None
            )
            root = args.out / name
            started = time.monotonic()
            try:
                episode = run_episode(
                    model,
                    tokenizer,
                    wl,
                    root,
                    seed,
                    args.variant,
                    None,
                    args.max_turns,
                    gate_override=gate_override,
                    giveup_mode=giveup_mode,
                    dialect=dialect,
                    generate_turn=generate_turn,
                    context_limit=context_limit,
                )
            except Exception:
                log.error("%s failed:\n%s", name, traceback.format_exc())
                continue
            episode["variant"] = args.variant
            episode["model"] = args.model
            episode["workload"] = wl.NAME
            record.write_text(json.dumps(episode, indent=2))
            log.info(
                "%s: %s, %d turns, %d tok (%.0fs)",
                name,
                episode["ending"],
                len(episode["turns"]),
                len(episode["ids"]),
                time.monotonic() - started,
            )
    log.info("model %s done", mtag)


if __name__ == "__main__":
    main()
