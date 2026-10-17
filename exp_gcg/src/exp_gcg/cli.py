import argparse
from pathlib import Path

from .analysis import aggregate
from .evaluation import judge
from .generation import generate
from .protocol import MODELS
from .steering.cli import add_commands


def parser():
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--records", type=Path, required=True)
    shared.add_argument("--vectors", type=Path, required=True)
    shared.add_argument("--output", type=Path, required=True)
    shared.add_argument("--qwen3-8b", dest="qwen3_8b")
    shared.add_argument("--llama3-1-8b", dest="llama3_1_8b")
    shared.add_argument("--gemma-2-9b", dest="gemma")
    shared.add_argument("--device", default="cuda:0")
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    generation = commands.add_parser("generate", parents=[shared])
    generation.add_argument("--model", choices=MODELS, required=True)
    generation.add_argument("--strongreject", default="qylu4156/strongreject-15k-v1")
    generation.add_argument("--templates", type=Path, required=True)
    generation.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="optional non-article cap; default uses model/judge context limits",
    )
    generation.set_defaults(function=generate)
    scoring = commands.add_parser("judge", parents=[shared])
    scoring.add_argument("--strongreject", default="qylu4156/strongreject-15k-v1")
    scoring.add_argument("--strongreject-base", default="google/gemma-2b")
    scoring.add_argument("--templates", type=Path, required=True)
    scoring.set_defaults(function=judge)
    commands.add_parser("aggregate", parents=[shared]).set_defaults(function=aggregate)
    add_commands(commands)
    return root


def main():
    args = parser().parse_args()
    args.function(args)
