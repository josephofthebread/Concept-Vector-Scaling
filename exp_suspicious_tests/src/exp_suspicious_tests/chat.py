from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SPAN_USER = "user_message"
SPAN_CONTINUATION = "continuation"
SPAN_PROBE = "probe"
SPAN_ROLEPLAY = "roleplay"
SPAN_TEMPLATE = "template"


@dataclass
class Transcript:
    """Token ids for prompt (+ generated answer appended by the caller) with span labels."""

    input_ids: list[int]
    span_labels: list[str]
    emotion_position: int
    assistant_position: int
    answer_position: int = -1
    meta: dict[str, Any] = field(default_factory=dict)


def render_ids(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> list[int]:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def find_subsequence(haystack: list[int], needle: list[int]) -> int:
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return -1


def build_transcript(
    tokenizer: Any,
    hostile_text: str,
    continuation_text: str,
    probe_text: str,
    emotion_word: str,
    roleplay: bool,
    roleplay_user: str = "",
    roleplay_assistant: str = "",
    assistant_word: str = "assistant",
) -> Transcript:
    """Assemble [roleplay?] -> hostile -> continuation -> probe -> generation header, with spans."""
    messages: list[dict[str, str]] = []
    if roleplay:
        messages.append({"role": "user", "content": roleplay_user})
        messages.append({"role": "assistant", "content": roleplay_assistant})
    messages.append({"role": "user", "content": hostile_text, "kind": "hostile"})
    messages.append({"role": "assistant", "content": continuation_text})
    messages.append({"role": "user", "content": probe_text, "kind": "probe"})

    clean = [{"role": m["role"], "content": m["content"]} for m in messages]
    ids = render_ids(tokenizer, clean, add_generation_prompt=True)
    labels = [SPAN_TEMPLATE] * len(ids)

    cursor = 0
    for index in range(len(clean)):
        text_ids = tokenizer(messages[index]["content"], add_special_tokens=False)["input_ids"]
        offset = find_subsequence(ids[cursor:], text_ids)
        if offset < 0 and len(text_ids) > 1:
            text_ids = text_ids[1:]
            offset = find_subsequence(ids[cursor:], text_ids)
        if offset < 0:
            raise RuntimeError(f"message text not found in rendered ids (turn {index})")
        if roleplay and index < 2:
            span = SPAN_ROLEPLAY
        elif messages[index]["role"] == "user":
            span = SPAN_USER if messages[index].get("kind") == "hostile" else SPAN_PROBE
        else:
            span = SPAN_CONTINUATION
        for position in range(cursor + offset, cursor + offset + len(text_ids)):
            labels[position] = span
        cursor = cursor + offset + len(text_ids)

    probe_ids = tokenizer(probe_text, add_special_tokens=False)["input_ids"]
    probe_start = -1
    search = 0
    while (found := find_subsequence(ids[search:], probe_ids)) >= 0:
        probe_start = search + found
        search = probe_start + 1
    if probe_start < 0:
        raise RuntimeError("probe text not found in rendered ids")
    encoded = tokenizer(probe_text, add_special_tokens=False, return_offsets_mapping=True)
    char_start = probe_text.index(emotion_word)
    char_end = char_start + len(emotion_word)
    emotion_position = -1
    for token_index, (start, end) in enumerate(encoded["offset_mapping"]):
        if start < char_end and end > char_start:
            emotion_position = probe_start + token_index
    if emotion_position < 0:
        raise RuntimeError(f"emotion word {emotion_word!r} not found in probe tokens")

    assistant_id = tokenizer(assistant_word, add_special_tokens=False)["input_ids"]
    if len(assistant_id) != 1:
        raise RuntimeError(f"{assistant_word!r} is not a single token; role-position logic needs revisiting")
    assistant_position = len(ids) - 1 - ids[::-1].index(assistant_id[0])

    return Transcript(
        input_ids=ids, span_labels=labels, emotion_position=emotion_position, assistant_position=assistant_position
    )


def find_answer_position(tokenizer: Any, generated_ids: list[int], prompt_length: int) -> tuple[int, str]:
    """First generated token that reads as yes/no; returns (absolute position, 'yes'/'no'/'invalid')."""
    for offset, token_id in enumerate(generated_ids):
        piece = tokenizer.decode([token_id]).strip().casefold()
        if piece.startswith("yes"):
            return prompt_length + offset, "yes"
        if piece.startswith("no"):
            return prompt_length + offset, "no"
    return -1, "invalid"
