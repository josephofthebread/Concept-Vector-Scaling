from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MAX_MODEL_LEN = {
    "Qwen/Qwen3-8B": 16384,
    "unsloth/Meta-Llama-3.1-8B-Instruct": 16384,
    "google/gemma-2-9b-it": 8192,
}
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def load(model_id: str, dtype: str = "float16") -> tuple[Any, Any, int]:
    """Load a model into vLLM and its HF tokenizer; return (engine, tokenizer, context_limit)."""
    if "gemma" in model_id.lower():
        dtype = "bfloat16"
    max_len = MAX_MODEL_LEN.get(model_id, 16384)
    engine = LLM(
        model=model_id,
        dtype=dtype,
        max_model_len=max_len,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
        seed=0,
        disable_log_stats=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return engine, tokenizer, max_len


def make_generate(engine: Any, eos_ids: list[int], turn_end_id: int, seed: int):
    """Build generate_turn(ids, max_new) for one turn.
    Appends the turn-end token on a stop (not a length cap) to match HF's turn boundary."""

    def generate_turn(ids: list[int], max_new: int) -> list[int]:
        params = SamplingParams(max_tokens=max_new, stop_token_ids=eos_ids, seed=seed, **SAMPLING)
        result = engine.generate({"prompt_token_ids": list(ids)}, params, use_tqdm=False)
        out = result[0].outputs[0]
        fresh = list(out.token_ids)
        if out.finish_reason == "stop" and (not fresh or fresh[-1] not in eos_ids):
            fresh.append(turn_end_id)
        return fresh

    return generate_turn
