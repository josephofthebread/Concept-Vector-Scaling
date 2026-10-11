from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from exp_attn_interp.config import (
    INSTRUCT_MODEL,
    MIN_CONTEXT,
    READ_LAYER,
    SINK_COSINE,
    SINK_EPSILON,
    SINK_LEADING_TOKENS,
    SINK_NORM_MULT,
)

TURN_START = "<|im_start|>"
TURN_END = "<|im_end|>"
ASSISTANT_WORD = "assistant"


def load_model(offline: bool = True, model_name: str = INSTRUCT_MODEL) -> tuple[Any, Any]:
    """Load an instruct checkpoint with eager attention (sdpa drops attentions and gemma-2's soft-cap)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=offline, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=offline,
        device_map={"": "cuda"},
    ).eval()
    return model, tokenizer


def find_subsequence(haystack: list[int], needle: list[int]) -> int:
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return -1


def story_span(tokenizer: Any, ids: list[int], story: str) -> tuple[int, int]:
    """Return the [start, end) span of the story's own tokens inside a rendered chat sequence."""
    needle = tokenizer(story, add_special_tokens=False)["input_ids"]
    off = find_subsequence(ids, needle)
    if off < 0 and len(needle) > 1:
        needle = needle[1:]
        off = find_subsequence(ids, needle)
    if off < 0:
        raise RuntimeError("story tokens not found in rendered sequence")
    return off, off + len(needle)


class ValueCatcher:
    """Hooks every layer's v_proj and turns values into per-query-head output norms."""

    def __init__(self, model: Any) -> None:
        config = model.config
        self.heads = int(config.num_attention_heads)
        self.kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", config.hidden_size // self.heads))
        self.values: list[torch.Tensor | None] = [None] * len(model.layers)
        self.grams: list[torch.Tensor] = []
        for index, block in enumerate(model.layers):
            block.self_attn.v_proj.register_forward_hook(self._catch(index))
            weight = block.self_attn.o_proj.weight
            slices = weight.view(weight.shape[0], self.heads, self.head_dim).to(torch.float32)
            self.grams.append(torch.einsum("ohd,ohe->hde", slices, slices))

    def _catch(self, index: int) -> Any:
        def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            self.values[index] = output.detach()

        return hook

    def value_norms(self, index: int) -> torch.Tensor:
        """[heads, T] -- ||W_O_head @ v_kv(head)(t)|| via the per-head Gram matrix."""
        raw = self.values[index]
        if raw is None:
            raise RuntimeError(f"v_proj hook missed layer {index}")
        tokens = raw.shape[1]
        values = raw.view(tokens, self.kv_heads, self.head_dim).to(torch.float32)
        group = self.heads // self.kv_heads
        gram = self.grams[index].to(raw.device)
        norms = torch.empty(self.heads, tokens, device=raw.device, dtype=torch.float32)
        for head in range(self.heads):
            v = values[:, head // group]
            quad = torch.einsum("td,de,te->t", v, gram[head], v)
            norms[head] = quad.clamp_min(0).sqrt()
        return norms


def sink_mask(attentions: list[torch.Tensor], hidden: list[torch.Tensor]) -> torch.Tensor:
    """Mask [T] of leading tokens, sink-like high-attention tokens and residual-norm outliers."""
    tokens = attentions[0].shape[-1]
    device = attentions[0].device
    mask = torch.zeros(tokens, dtype=torch.bool, device=device)
    mask[:SINK_LEADING_TOKENS] = True
    for index, attn in enumerate(attentions):
        received = attn[0].mean(dim=0).mean(dim=0)
        candidates = (received > SINK_EPSILON).nonzero().flatten()
        state = hidden[index][0].to(torch.float32)
        primary = torch.nn.functional.normalize(state[0], dim=0)
        for token in candidates.tolist():
            if token < SINK_LEADING_TOKENS:
                continue
            if torch.nn.functional.normalize(state[token], dim=0) @ primary > SINK_COSINE:
                mask[token] = True
    norms = hidden[READ_LAYER][0].to(torch.float32).norm(dim=-1)
    threshold = SINK_NORM_MULT * norms.median()
    mask |= norms > threshold
    return mask


def build_positions(tokenizer: Any, ids: list[int], prompt_length: int, grid_positions: int) -> list[tuple[int, str]]:
    """(position, kind): named special positions (role tokens, im_end, reply start, last) + grid."""
    total = len(ids)
    im_start = tokenizer(TURN_START, add_special_tokens=False)["input_ids"][0]
    im_end = tokenizer(TURN_END, add_special_tokens=False)["input_ids"][0]
    special: list[tuple[int, str]] = []
    header_roles = [index + 1 for index, token in enumerate(ids[:-1]) if token == im_start]
    if header_roles:
        special.append((header_roles[0], "role_user"))
    if len(header_roles) > 1:
        special.append((header_roles[-1], "role_assistant"))
    special += [(index, "im_end") for index, token in enumerate(ids) if token == im_end]
    if prompt_length < total:
        special.append((prompt_length, "reply_first"))
    special.append((total - 1, "last"))

    taken = {position for position, _ in special}
    grid = torch.linspace(MIN_CONTEXT, total - 2, grid_positions).round().long().tolist()
    positions = [(p, kind) for p, kind in special if MIN_CONTEXT <= p < total]
    positions += [(p, "grid") for p in sorted(set(grid)) if p not in taken and MIN_CONTEXT <= p < total]
    positions.sort(key=lambda pair: pair[0])
    return positions


def token_cosine(hidden_layer: torch.Tensor, vectors_layer: torch.Tensor) -> torch.Tensor:
    """cos[T, pairs] between each token's unit residual and each unit concept vector."""
    unit = torch.nn.functional.normalize(hidden_layer.to(torch.float32), dim=-1)
    return unit @ vectors_layer.T


def center(cos: torch.Tensor, mode: str) -> torch.Tensor:
    """Z-score a [T, pairs] cosine matrix: raw, per concept over tokens, per token over concepts,
    or double (concept then token)."""
    if mode == "raw":
        return cos
    if mode == "concept":
        return _zscore(cos, dim=0)
    if mode == "token":
        return _zscore(cos, dim=1)
    if mode == "double":
        return _zscore(_zscore(cos, dim=0), dim=1)
    raise ValueError(f"unknown centering {mode!r}")


def _zscore(x: torch.Tensor, dim: int) -> torch.Tensor:
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True).clamp_min(1e-6)
    return (x - mean) / std


def generate_reply(
    model: Any, tokenizer: Any, text: str, item_id: int, reply_tokens: int, sampling: dict[str, Any], seed: int
) -> tuple[list[int], list[int]]:
    """Chat prompt from a first user message; a sampled reply, deterministic in (item_id, seed)."""
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    torch.manual_seed(seed)
    batch = torch.tensor([prompt_ids], device=model.device)
    with torch.inference_mode():
        generated = model.generate(
            batch,
            attention_mask=torch.ones_like(batch),
            max_new_tokens=reply_tokens,
            pad_token_id=tokenizer.eos_token_id,
            **sampling,
        )
    reply_ids = generated[0, len(prompt_ids) :].tolist()
    return prompt_ids, reply_ids
