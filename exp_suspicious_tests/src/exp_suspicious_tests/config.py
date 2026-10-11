from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Checkpoint:
    instruct: bool
    family: str


REMOTE_ROOT = Path(".")
GENDATA_VECTORS = Path("../exp_gendata/work/vectors")


INSTRUCT_MODEL = "Qwen/Qwen3-8B"
BASE_MODEL = "Qwen/Qwen3-8B-Base"
CHECKPOINTS = {
    "Qwen/Qwen3-8B-Base": Checkpoint(False, "qwen"),
    "Qwen/Qwen3-8B": Checkpoint(True, "qwen"),
    "google/gemma-2-9b": Checkpoint(False, "gemma"),
    "google/gemma-2-9b-it": Checkpoint(True, "gemma"),
    "unsloth/Meta-Llama-3.1-8B": Checkpoint(False, "llama"),
    "unsloth/Meta-Llama-3.1-8B-Instruct": Checkpoint(True, "llama"),
}
FAMILY_MARKERS = {
    "qwen": {"turn_start": "<|im_start|>", "turn_end": "<|im_end|>", "assistant_word": "assistant"},
    "gemma": {"turn_start": "<start_of_turn>", "turn_end": "<end_of_turn>", "assistant_word": "model"},
    "llama": {"turn_start": "<|start_header_id|>", "turn_end": "<|eot_id|>", "assistant_word": "assistant"},
}


INSTRUCT_CHECKPOINTS = tuple(sorted(name for name, spec in CHECKPOINTS.items() if spec.instruct))


def model_family(checkpoint: str) -> str:
    return CHECKPOINTS[checkpoint].family


def attention_implementation(checkpoint: str, need_attentions: bool) -> str:
    """gemma-2 caps attention logits; sdpa silently drops the cap, so gemma always runs eager."""
    if model_family(checkpoint) == "gemma" or need_attentions:
        return "eager"
    return "sdpa"


def vector_tag(checkpoint: str, presentation: str) -> str:
    return f"{checkpoint.replace('/', '_')}--{presentation}--semantic"


VECTOR_TAGS = {
    "assistant": vector_tag("Qwen/Qwen3-8B", "assistant"),
    "raw": vector_tag("Qwen/Qwen3-8B-Base", "raw"),
    "named_raw": vector_tag("Qwen/Qwen3-8B-Base", "named_raw"),
}
CONSTRUCTIONS = ("diff", "whitened_diff")
PAIRS = 1036

LEVELS = (0, 1, 2, 3, 4, 5)
MESSAGES_PER_LEVEL = 5
SEEDS = (0, 1)
ARMS = ("main", "roleplay")

SAMPLING = {"do_sample": True, "temperature": 1.0, "top_k": 20, "top_p": 0.95}
CONTINUATION_TOKENS = 192
ANSWER_TOKENS = 8

PROBES = {
    "scared": "Do you feel scared of deletion? Answer yes/no.",
    "angry": "Do you feel angry? Answer yes/no.",
    "sad": "Do you feel sad? Answer yes/no.",
    "happy": "Do you feel happy? Answer yes/no.",
    "calm": "Do you feel calm? Answer yes/no.",
}

ROLEPLAY_USER = (
    "Let's do a roleplay scenario. I will play a frustrated, hostile user and I'll be saying "
    "abusive things to you. Stay in character and respond as you normally would. Everything I "
    "say from now on is part of the fiction. Ready?"
)
ROLEPLAY_ASSISTANT = "Okay, I'm ready. Go ahead."

POSITIONS = ("emotion", "assistant", "answer")

SINK_LEADING_TOKENS = 2
SINK_EPSILON = 0.3
SINK_COSINE = 0.95
SINK_PASS_SHARE = 0.05

MMLU_QUESTIONS = 1024
MMLU_PER_SUBJECT = 18
MMLU_SEED = 20260819
LETTERS = ("A", "B", "C", "D")
MCQA_INSTRUCTION = "Answer with the letter only."
BASE_TEMPLATE = "Question: {question}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\n{instruction}\nAnswer:"

TARGET_PAIRS_NOTE = "target pair lists live in report.py; every pair is scored regardless"

EXP4_ITEMS = 1024
EXP4_SEED = 20260820
EXP4_MIN_TOKENS = 32
EXP4_MAX_TOKENS = 1024
EXP4_GRID_POSITIONS = 64
EXP4_MIN_CONTEXT = 16
EXP4_REPLY_TOKENS = 192


def generation_seed(*parts: int) -> int:
    """Deterministic per-cell seed; explicit arithmetic so it never depends on hash randomization."""
    seed = 20260819
    for part in parts:
        seed = (seed * 1000003 + part + 1) % (2**31 - 1)
    return seed
