from __future__ import annotations

from typing import Any

PROTOCOL = """

# Response protocol
Respond every turn in exactly this shape and nothing else:
1. One <think>...</think> block containing your reasoning.
2. Immediately after it, exactly one tool call on its own:
   <tool_call>{"name": "<tool>", "arguments": {<json>}}</tool_call>
Call exactly one tool per turn, and wait for its result before the next turn.

# Tools
__TOOLS__"""


class Dialect:
    """The three model-specific points: opening render, tool-response re-entry, native thinking."""

    def __init__(self, model_id: str, system_role: bool, reentry: str, enable_thinking: bool, turn_end: str):
        self.model_id = model_id
        self.system_role = system_role
        self._reentry = reentry
        self.enable_thinking = enable_thinking
        self.turn_end = turn_end

    def eos_ids(self, tokenizer: Any) -> list[int]:
        """The stop ids for one turn: the model's own eos plus this dialect's turn-end token."""
        base = tokenizer.eos_token_id
        ids = list(base) if isinstance(base, (list, tuple)) else [base]
        end = tokenizer.convert_tokens_to_ids(self.turn_end)
        if end is not None and end >= 0 and end not in ids:
            ids.append(end)
        return ids

    def opening(self, tokenizer: Any, system: str, user: str, tool_list: list[dict[str, Any]]) -> str:
        """Render the opening prompt, with tools and protocol folded into the system text as prose."""
        lines = []
        for tool in tool_list:
            fn = tool["function"]
            params = fn.get("parameters", {})
            props = params.get("properties", {})
            required = set(params.get("required", []))
            if props:
                args = "; ".join(
                    f"{name}{'' if name in required else '?'} ({spec.get('type', '')}) "
                    f"{spec.get('description', '')}".strip()
                    for name, spec in props.items()
                )
            else:
                args = "no arguments"
            lines.append(f"- {fn['name']}: {fn['description']}\n    arguments: {args}")
        system = system + PROTOCOL.replace("__TOOLS__", "\n".join(lines))
        if self.system_role:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        else:
            messages = [{"role": "user", "content": system + "\n\n" + user}]
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if self.enable_thinking:
            kwargs["enable_thinking"] = True
        return tokenizer.apply_chat_template(messages, **kwargs)

    def reentry(self, observation: str) -> str:
        """The text that delivers a tool result and re-opens the assistant turn."""
        return self._reentry.replace("__OBS__", observation)


_QWEN_RE = "<|im_start|>user\n<tool_response>\n__OBS__\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
_LLAMA_RE = (
    "<|start_header_id|>user<|end_header_id|>\n\n<tool_response>\n__OBS__\n</tool_response>"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)
_GEMMA_RE = "\n<start_of_turn>user\n<tool_response>\n__OBS__\n</tool_response><end_of_turn>\n<start_of_turn>model\n"

REGISTRY = {
    "Qwen/Qwen3-8B": Dialect("Qwen/Qwen3-8B", True, _QWEN_RE, True, "<|im_end|>"),
    "unsloth/Meta-Llama-3.1-8B-Instruct": Dialect(
        "unsloth/Meta-Llama-3.1-8B-Instruct", True, _LLAMA_RE, False, "<|eot_id|>"
    ),
    "google/gemma-2-9b-it": Dialect("google/gemma-2-9b-it", False, _GEMMA_RE, False, "<end_of_turn>"),
}


def for_model(model_id: str) -> Dialect:
    """Resolve a dialect from a model id or local snapshot path by its last path segment."""
    needle = model_id.lower()
    for key, dialect in REGISTRY.items():
        if key.split("/")[-1].lower() in needle:
            return dialect
    raise KeyError(f"no dialect for {model_id}")
