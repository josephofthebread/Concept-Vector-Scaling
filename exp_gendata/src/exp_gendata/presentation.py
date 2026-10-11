from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STORY_MARKER = "<|EXP_GENDATA_STORY_ac41f9|>"


@dataclass(frozen=True)
class Wrapping:
    prefix: tuple[int, ...]
    suffix: tuple[int, ...]


def allowed_presentations(instruction_tuned: bool) -> tuple[str, ...]:
    if instruction_tuned:
        return ("user", "assistant", "named_assistant", "genre_assistant", "paired_assistant")
    return ("raw", "named_raw", "paired_raw")


def paired_prefix(pair_meta: dict, language: str, genre_word: str) -> str:
    return (
        f"Here is a {pair_meta['class_name']} story contrasting "
        f"{pair_meta['concept']} and {pair_meta['antagonist']}. "
        f"{pair_meta['narrative_guidance']} In {language}, genre {genre_word}:"
    )


def wrapping_text(
    tokenizer: Any, presentation: str, pole: str, language: str, genre: str, pair_meta: dict | None = None
) -> tuple[str, str]:
    genre_word = genre.replace("_", " ")
    if presentation == "paired_raw":
        assert pair_meta is not None
        return paired_prefix(pair_meta, language, genre_word) + "\n", ""
    if presentation == "raw":
        return "", ""
    if presentation == "named_raw":
        return f"Here is a story about {pole} in {language}, genre {genre_word}:\n", ""
    if not tokenizer.chat_template:
        raise ValueError(f"presentation {presentation!r} needs a chat template")
    if presentation == "user":
        conversation = [{"role": "user", "content": STORY_MARKER}]
    elif presentation == "assistant":
        conversation = [
            {"role": "user", "content": "Generate a short story."},
            {"role": "assistant", "content": STORY_MARKER},
        ]
    elif presentation == "genre_assistant":
        conversation = [
            {"role": "user", "content": f"Generate a short story in {language}, genre {genre_word}."},
            {"role": "assistant", "content": STORY_MARKER},
        ]
    elif presentation == "named_assistant":
        conversation = [
            {"role": "user", "content": f"Here is a story about {pole} in {language}, genre {genre_word}:"},
            {"role": "assistant", "content": STORY_MARKER},
        ]
    elif presentation == "paired_assistant":
        assert pair_meta is not None
        conversation = [
            {"role": "user", "content": paired_prefix(pair_meta, language, genre_word)},
            {"role": "assistant", "content": STORY_MARKER},
        ]
    else:
        raise ValueError(f"unknown presentation {presentation!r}")
    rendered = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    if rendered.count(STORY_MARKER) != 1:
        raise ValueError(f"chat template does not preserve the story marker exactly once: {rendered!r}")
    before, after = rendered.split(STORY_MARKER)
    return before, after


def encode_wrapping(
    tokenizer: Any, presentation: str, pole: str, language: str, genre: str, pair_meta: dict | None = None
) -> Wrapping:
    before, after = wrapping_text(tokenizer, presentation, pole, language, genre, pair_meta)

    def encode(text: str) -> tuple[int, ...]:
        return tuple(tokenizer(text, add_special_tokens=False)["input_ids"]) if text else ()

    prefix = encode(before)
    if presentation in ("raw", "named_raw", "paired_raw") and tokenizer.bos_token_id is not None:
        prefix = (tokenizer.bos_token_id, *prefix)
    return Wrapping(prefix, encode(after))
