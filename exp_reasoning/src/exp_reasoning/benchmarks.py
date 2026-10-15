from __future__ import annotations

import os
import random
import re
from typing import Any

import torch
from datasets import load_dataset
from transformers import (
    StoppingCriteria,
)

MODEL_ID = "Qwen/Qwen3-8B"


class UniqueNGramLoopDetector(StoppingCriteria):
    """Stop after sustained low unique-n-gram diversity, preserving a tail for analysis."""

    def __init__(
        self,
        prompt_tokens: int,
        *,
        ngram_size: int,
        window_tokens: int,
        unique_ratio_threshold: float,
        check_every: int,
        consecutive_windows: int,
        min_new_tokens: int,
        extra_tokens: int,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.ngram_size = ngram_size
        self.window_tokens = window_tokens
        self.unique_ratio_threshold = unique_ratio_threshold
        self.check_every = check_every
        self.consecutive_windows = consecutive_windows
        self.min_new_tokens = min_new_tokens
        self.extra_tokens = extra_tokens

        self.last_checked_tokens = 0
        self.low_ratio_checks = 0
        self.last_unique_ratio: float | None = None
        self.detected = False
        self.forced_stop = False
        self.window_start_token: int | None = None
        self.detection_token: int | None = None
        self.detection_unique_ratio: float | None = None
        self.stop_after_token: int | None = None

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> torch.BoolTensor:
        del scores, kwargs
        generated_tokens = int(input_ids.shape[1] - self.prompt_tokens)

        if self.detected:
            if self.stop_after_token is not None and generated_tokens >= self.stop_after_token:
                self.forced_stop = True
                return torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        if generated_tokens < max(self.min_new_tokens, self.window_tokens):
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if generated_tokens - self.last_checked_tokens < self.check_every:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        self.last_checked_tokens = generated_tokens

        window = input_ids[0, -self.window_tokens :].detach().cpu().tolist()
        total_ngrams = len(window) - self.ngram_size + 1
        if total_ngrams <= 0:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        unique_ngrams = len({tuple(window[index : index + self.ngram_size]) for index in range(total_ngrams)})
        ratio = unique_ngrams / total_ngrams
        self.last_unique_ratio = ratio

        if ratio < self.unique_ratio_threshold:
            self.low_ratio_checks += 1
        else:
            self.low_ratio_checks = 0

        if self.low_ratio_checks >= self.consecutive_windows:
            self.detected = True
            self.window_start_token = max(0, generated_tokens - self.window_tokens)
            self.detection_token = generated_tokens
            self.detection_unique_ratio = ratio
            self.stop_after_token = generated_tokens + self.extra_tokens

        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    def metadata(self, final_generated_tokens: int) -> dict[str, Any]:
        preserved_tail = (
            max(0, final_generated_tokens - self.detection_token) if self.detection_token is not None else 0
        )
        return {
            "enabled": True,
            "detected": self.detected,
            "forced_stop": self.forced_stop,
            "ngram_size": self.ngram_size,
            "window_tokens": self.window_tokens,
            "unique_ratio_threshold": self.unique_ratio_threshold,
            "check_every_tokens": self.check_every,
            "consecutive_windows": self.consecutive_windows,
            "min_new_tokens": self.min_new_tokens,
            "configured_extra_tokens": self.extra_tokens,
            "window_start_token": self.window_start_token,
            "detection_token": self.detection_token,
            "stop_token": final_generated_tokens,
            "preserved_tail_tokens": preserved_tail,
            "detection_unique_ngram_ratio": self.detection_unique_ratio,
            "last_unique_ngram_ratio": self.last_unique_ratio,
        }


def extract_choice(text: str) -> str | None:
    candidates = re.findall(r"(?:final\s+answer|answer)\s*(?:is|:)?\s*\**([A-D])\b", text, re.I)
    if not candidates:
        candidates = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", text)
    return candidates[-1].upper() if candidates else None


def load_benchmark(name: str) -> list[dict[str, Any]]:
    if name == "math_500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [{"id": str(i), "prompt": row["problem"], "answer": row["solution"]} for i, row in enumerate(dataset)]
    if name == "gpqa_diamond":
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        rows = []
        for i, row in enumerate(dataset):
            options = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            random.Random(i).shuffle(options)
            correct = "ABCD"[options.index(row["Correct Answer"])]
            rendered = "\n".join(f"{letter}. {option}" for letter, option in zip("ABCD", options, strict=True))
            rows.append({"id": str(i), "prompt": f"{row['Question']}\n\n{rendered}", "answer": correct})
        return rows
    raise ValueError(name)


def instruction(name: str, question: str) -> str:
    if name == "math_500":
        return (
            "Solve the following math problem efficiently and clearly. The last line "
            "of your response should be of the following format: 'Therefore, the final "
            "answer is: $\\boxed{ANSWER}$. I hope it is correct' (without quotes) "
            "where ANSWER is just the final number or expression that solves the "
            f"problem. Think step by step before answering.\n\n{question}"
        )
    suffix = {
        "gpqa_diamond": "Reason carefully, then end with only the letter of the correct option (A, B, C, or D).",
    }[name]
    return f"Solve the following problem. {suffix}\n\n{question}"


def score(name: str, output: str, answer: str) -> bool | None:
    if name == "gpqa_diamond":
        return extract_choice(output) == answer
    if name == "math_500":
        return math_equal(output, answer)
    raise ValueError(name)


def visible_gpu_ids() -> list[str]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is not None:
        return [device.strip() for device in configured.split(",") if device.strip()]
    return [str(index) for index in range(torch.cuda.device_count())]


def math_equal(prediction: str, reference: str) -> bool | None:
    """Compare a generated MATH answer with its reference using math-verify."""
    try:
        from math_verify import LatexExtractionConfig, parse, verify
    except ImportError:
        return None
    try:
        gold = parse(reference, extraction_config=[LatexExtractionConfig()])
        pred = parse(prediction, extraction_config=[LatexExtractionConfig()])
        return bool(verify(gold, pred))
    except Exception:
        return False


def _subsequence(sequence: list[int], needle: list[int]) -> int | None:
    if not needle:
        return None
    for index in range(len(sequence) - len(needle) + 1):
        if sequence[index : index + len(needle)] == needle:
            return index
    return None


def thinking_span(tokenizer: Any, token_ids: list[int], available: int) -> tuple[int | None, int, str]:
    opening = tokenizer.encode("<think>", add_special_tokens=False)
    closing = tokenizer.encode("</think>", add_special_tokens=False)
    start = _subsequence(token_ids[:available], opening)
    if start is None:
        return None, 0, "no_thinking_tags"
    start += len(opening)
    finish = _subsequence(token_ids[start:available], closing)
    if finish is None:
        return start, available, "unclosed_thinking"
    return start, start + finish, "closed_thinking"
