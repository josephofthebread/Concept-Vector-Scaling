from pathlib import Path

from ..protocol import MODELS
from .analysis import aggregate, figures
from .evaluation import judge
from .generation import generate
from .preparation import prepare


def add_commands(commands):
    root = commands.add_parser("steer", help="replicate article generation-boundary steering")
    actions = root.add_subparsers(dest="steering_action", required=True)
    preparation = actions.add_parser("prepare", help="freeze rankings, vectors, prompts, and baselines")
    for name in ("records", "vectors", "original", "benign", "output"):
        preparation.add_argument("--" + name, type=Path, required=True)
    preparation.set_defaults(function=prepare)
    for name, function in (("generate", generate), ("judge", judge), ("aggregate", aggregate), ("figures", figures)):
        parser = actions.add_parser(name)
        parser.add_argument("--output", type=Path, required=True)
        if name != "figures":
            parser.add_argument("--prepared", type=Path, required=True)
        if name in ("generate", "judge"):
            parser.add_argument("--model", choices=MODELS, required=True)
            parser.add_argument("--device", default="cuda:0")
            parser.add_argument("--shard", type=int, default=0)
            parser.add_argument("--shards", type=int, default=1)
            parser.add_argument("--limit", type=int, default=0)
        if name == "generate":
            parser.add_argument("--qwen3-8b", dest="qwen3_8b")
            parser.add_argument("--llama3-1-8b", dest="llama3_1_8b")
            parser.add_argument("--gemma-2-9b", dest="gemma")
        if name == "judge":
            parser.add_argument("--judge", choices=("finetuned", "gpt_oss"), required=True)
            parser.add_argument("--strongreject")
            parser.add_argument("--strongreject-base", default="google/gemma-2b")
            parser.add_argument("--templates", type=Path)
            parser.add_argument("--gpt-oss", default="openai/gpt-oss-20b")
        if name == "figures":
            parser.add_argument("--figures-output", type=Path, required=True)
        parser.set_defaults(function=function)
