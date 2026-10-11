from __future__ import annotations

from pathlib import Path

MODEL = "Qwen/Qwen3-8B"

VEC_ROOT = Path("../exp_gendata/work/vectors")
VEC_TAG = {
    "Qwen/Qwen3-8B": "Qwen_Qwen3-8B--assistant--semantic",
    "Qwen/Qwen3-8B-Base": "Qwen_Qwen3-8B-Base--raw--semantic",
    "google/gemma-2-9b-it": "google_gemma-2-9b-it--assistant--semantic",
    "google/gemma-2-9b": "google_gemma-2-9b--raw--semantic",
    "unsloth/Meta-Llama-3.1-8B-Instruct": "unsloth_Meta-Llama-3.1-8B-Instruct--assistant--semantic",
    "unsloth/Meta-Llama-3.1-8B": "unsloth_Meta-Llama-3.1-8B--raw--semantic",
}

DEPTH_LO, DEPTH_HI = 1.0 / 3.0, 2.0 / 3.0


def swept_layers(num_hidden_layers: int) -> list[int]:
    """The inclusive integer layer band for a model of the given depth."""
    return list(range(round(DEPTH_LO * num_hidden_layers), round(DEPTH_HI * num_hidden_layers) + 1))


N_ROWS = 128000
SHARDS = 8
CAP = 256
MIN_TOKENS = 8
TOPK = 128

GARBAGE_FILTER = True
MIN_UNIQUE_RATIO = 0.4
MIN_ALPHA_FRAC = 0.5

LEVELS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
UPPER = [5, 6, 7, 8]
LOWER = [3, 2, 1, 0]
POLE_WEIGHTS = [1.0, 1.5, 2.0, 2.5]

BATCH = 256


def batch_for(device_name: str) -> int:
    """Sequences per forward; uniform across devices."""
    return BATCH
