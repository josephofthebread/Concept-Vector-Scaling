from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from .generation import (
    MAX_VALIDATION_RETRIES,
    build_generation_tasks,
    build_story_prompt,
    generate_tasks,
)
from .ontology import iter_feature_pairs, load_ontology
from .pairing import pair_stories_jsonl

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PACKAGE_ROOT / p


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Config must be a mapping: {path}")
    return cfg


def model_slug(model: str) -> str:
    return model.replace("/", "__").replace(".", "_")


def select_pairs(cfg: dict, class_id: int | None, pair_index: int | None):
    pairs = iter_feature_pairs(load_ontology(resolve(cfg["ontology_path"])))
    if class_id is not None:
        pairs = [p for p in pairs if p.class_id == class_id]
    if pair_index is not None:
        pairs = [p for p in pairs if p.pair_index == pair_index]
    if not pairs:
        raise SystemExit("No ontology pairs matched the filters")
    return pairs


def make_tasks(cfg: dict, args: argparse.Namespace):
    gen = cfg["generation"]
    genres = [args.genre] if args.genre else list(cfg["genres"])
    languages = [args.language] if args.language else list(cfg["languages"])
    pairs = select_pairs(cfg, args.class_id, args.pair_index)
    tasks = build_generation_tasks(
        pairs,
        genres,
        languages,
        stories_per_pole=int(gen["stories_per_pole"]),
        target_words_min=int(gen["target_words_min"]),
        target_words_max=int(gen["target_words_max"]),
    )
    if args.limit is not None:
        tasks = tasks[: args.limit]
    return tasks


def cmd_count(args: argparse.Namespace) -> None:
    cfg = load_config(resolve(args.config))
    pairs = iter_feature_pairs(load_ontology(resolve(cfg["ontology_path"])))
    classes = {p.class_id for p in pairs}
    genres = cfg["genres"]
    languages = cfg["languages"]
    spp = int(cfg["generation"]["stories_per_pole"])
    models = cfg["models"]
    cells_per_model = len(pairs) * len(genres) * len(languages)
    stories_per_model = cells_per_model * spp * 2
    print(f"classes: {len(classes)}")
    print(f"pairs: {len(pairs)}")
    print(f"genres: {len(genres)}")
    print(f"languages: {len(languages)}")
    print(f"setups per cell: {spp}")
    print(f"cells per generator: {cells_per_model}")
    print(f"narratives per generator: {stories_per_model}")
    print(f"generators: {len(models)}")
    print(f"narratives total: {stories_per_model * len(models)}")
    print(f"matched pair-rows total: {cells_per_model * spp * len(models)}")


def cmd_dump_prompt(args: argparse.Namespace) -> None:
    cfg = load_config(resolve(args.config))
    tasks = make_tasks(cfg, args)
    if not tasks:
        raise SystemExit("No tasks to dump")
    task = tasks[0]
    messages = build_story_prompt(task)
    payload = {
        "request_id": task.request_id,
        "concept": task.feature_pair.concept,
        "antagonist": task.feature_pair.antagonist,
        "class_name": task.feature_pair.class_name,
        "genre": task.genre,
        "language": task.language,
        "messages": messages,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out = resolve(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text, end="")


def cmd_generate(args: argparse.Namespace) -> None:
    cfg = load_config(resolve(args.config))
    gen = cfg["generation"]
    model = args.model or cfg["models"][0]
    tasks = make_tasks(cfg, args)
    out_dir = resolve(args.out_dir) if args.out_dir else resolve(cfg["output_dir"]) / model_slug(model)
    print(f"model={model}")
    print(f"cells={len(tasks)} out={out_dir}")
    asyncio.run(
        generate_tasks(
            tasks,
            out_dir=out_dir,
            model=model,
            temperature=float(gen["temperature"]),
            top_p=float(gen["top_p"]),
            max_tokens=int(gen["max_tokens"]),
            concurrency=int(args.concurrency if args.concurrency is not None else gen["concurrency"]),
            timeout_s=int(gen["request_timeout_s"]),
            max_retries=int(gen["max_retries"]),
            validation_retries=min(MAX_VALIDATION_RETRIES, int(gen["validation_retries"])),
            structured_outputs=bool(gen["structured_outputs"]),
            require_parameters=bool(gen["require_parameters"]),
            proxy=args.proxy,
            provider_sort=gen.get("provider_sort"),
            concurrency_adaptive=bool(gen.get("concurrency_adaptive", False)),
            concurrency_min=int(gen["concurrency_min"]) if gen.get("concurrency_min") is not None else None,
            concurrency_max=int(gen["concurrency_max"]) if gen.get("concurrency_max") is not None else None,
        )
    )


def cmd_pair(args: argparse.Namespace) -> None:
    cfg = load_config(resolve(args.config))
    model = args.model
    stories = (
        resolve(args.stories) if args.stories else resolve(cfg["output_dir"]) / model_slug(model) / "stories.jsonl"
    )
    out = resolve(args.out) if args.out else stories.with_name("paired.jsonl")
    n = pair_stories_jsonl(
        stories_path=stories,
        out_path=out,
        model=model,
        ontology_path=resolve(cfg["ontology_path"]),
    )
    print(f"wrote {n} paired rows to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exp_gendataset", description="Generate the Feature Stories corpus.")
    parser.add_argument("--config", default="data/paper.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    count = sub.add_parser("count", help="Print the paper matrix counts from the ontology and config")
    count.set_defaults(func=cmd_count)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--class-id", type=int, default=None)
    common.add_argument("--pair-index", type=int, default=None)
    common.add_argument("--genre", default=None)
    common.add_argument("--language", default=None)
    common.add_argument("--limit", type=int, default=None, help="Keep only the first N cells")

    dump = sub.add_parser("dump-prompt", parents=[common], help="Write the exact chat prompt for one cell")
    dump.add_argument("--out", default=None)
    dump.set_defaults(func=cmd_dump_prompt)

    gen = sub.add_parser("generate", parents=[common], help="Generate stories via OpenRouter")
    gen.add_argument("--model", default=None, help="OpenRouter model id; default is the first config model")
    gen.add_argument("--out-dir", default=None)
    gen.add_argument("--concurrency", type=int, default=None)
    gen.add_argument("--proxy", default=None)
    gen.set_defaults(func=cmd_generate)

    pair = sub.add_parser("pair", help="Join concept/antagonist stories into Hugging Face rows")
    pair.add_argument("--model", required=True)
    pair.add_argument("--stories", default=None)
    pair.add_argument("--out", default=None)
    pair.set_defaults(func=cmd_pair)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
