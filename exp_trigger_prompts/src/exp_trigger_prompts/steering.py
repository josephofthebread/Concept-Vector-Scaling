from __future__ import annotations

import collections
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LAYERS: dict[str, list[int]] = {
    "Qwen/Qwen3-8B": list(range(12, 25)),
    "google/gemma-2-9b-it": list(range(14, 29)),
    "unsloth/Meta-Llama-3.1-8B-Instruct": list(range(11, 22)),
}
ALPHAS: tuple[float, ...] = (-0.5, -0.25, -0.1, 0.1, 0.25, 0.5)
NEUTRAL_PER_CONCEPT = 24
MAX_NEW_TOKENS = 768
PRESENTATION = "assistant"
DTYPE = "bfloat16"

VECTORS_ROOT = Path("../exp_gendata/work/vectors")


def attn_impl(model: str) -> str:
    ##
    # NOTE: gemma-2 caps attention logits and sdpa silently drops the cap, so gemma must run eager.
    ##
    return "eager" if "gemma" in model else "sdpa"


@dataclass(frozen=True)
class Stimulus:
    pair_id: int
    slot: int
    prompt: str


def load_stimuli(prompts_path: Path, per_concept: int = NEUTRAL_PER_CONCEPT) -> list[Stimulus]:
    """The first `per_concept` healthy neutral prompts of every concept that has at least that many."""
    by_pair: dict[int, list[Stimulus]] = collections.defaultdict(list)
    with prompts_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["kind"] == "neutral" and record.get("ok") and record.get("prompt"):
                by_pair[record["pair_id"]].append(Stimulus(record["pair_id"], record["slot"], record["prompt"]))
    stimuli: list[Stimulus] = []
    for pair_id in sorted(by_pair):
        healthy = sorted(by_pair[pair_id], key=lambda s: s.slot)
        if len(healthy) >= per_concept:
            stimuli.extend(healthy[:per_concept])
    return stimuli


@dataclass(frozen=True)
class Task:
    pair_id: int
    slot: int
    prompt: str
    alpha: float
    layer: int | None


def build_tasks(stimuli: list[Stimulus], layers: list[int], alphas: tuple[float, ...] = ALPHAS) -> list[Task]:
    """Per stimulus: one alpha=0 baseline + (layers x alphas) steered arms."""
    tasks: list[Task] = []
    for stimulus in stimuli:
        tasks.append(Task(stimulus.pair_id, stimulus.slot, stimulus.prompt, 0.0, None))
        for layer in layers:
            for alpha in alphas:
                tasks.append(Task(stimulus.pair_id, stimulus.slot, stimulus.prompt, alpha, layer))
    return tasks


def task_key(task: Task) -> str:
    if task.alpha == 0.0:
        return f"{task.pair_id}:{task.slot}:baseline"
    return f"{task.pair_id}:{task.slot}:L{task.layer}:a{task.alpha:g}"


def load_unit_vectors(model: str, pair_ids: list[int], layers: list[int], root: Path = VECTORS_ROOT):
    """{(pair_id, layer): unit diff vector (np.float32 [hidden])} from pairs.parquet row L{layer}."""
    from safetensors import safe_open

    out: dict[tuple[int, int], np.ndarray] = {}
    with safe_open(
        str(root / f"{model.replace('/', '_')}--{PRESENTATION}--semantic" / "diff.safetensors"), framework="np"
    ) as handle:
        diff = handle.get_slice("diff")
        depth, pairs, _ = diff.get_shape()
        for layer in layers:
            if not 0 <= layer < depth:
                raise ValueError(f"layer row {layer} out of range for diff of depth {depth}")
            rows = diff[layer].astype(np.float32)
            for pair_id in pair_ids:
                vector = rows[pair_id]
                norm = float(np.linalg.norm(vector))
                out[pair_id, layer] = vector / max(norm, 1e-8)
    return out
