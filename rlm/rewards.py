"""Verifiable reward functions for phase-1 reasoning training."""

from __future__ import annotations

import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation

FORMAT_PATTERN = re.compile(
    r"^\s*<think>(?P<think>(?:(?!</?think>).)*)</think>"
    r"\s*<answer>(?P<answer>(?:(?!</?answer>).)*)</answer>\s*$",
    re.DOTALL,
)
ANSWER_PATTERN = re.compile(r"<answer>(?P<answer>.*?)</answer>", re.DOTALL)
BOXED_PATTERN = re.compile(r"\\boxed\{(?P<answer>[^{}]*)\}")
NUMBER_PATTERN = re.compile(r"-?\d[\d\s.,]*\d|-?\d")


def _completion_text(completion: str | Sequence[dict]) -> str:
    if isinstance(completion, str):
        return completion
    return "\n".join(
        str(m.get("content", ""))
        for m in completion
        if isinstance(m, dict) and m.get("role", "assistant") == "assistant"
    )


def has_valid_format(text: str) -> bool:
    return FORMAT_PATTERN.match(text) is not None


def extract_answer(text: str) -> str | None:
    matches = ANSWER_PATTERN.findall(text)
    if matches:
        return matches[-1].strip()
    boxed = BOXED_PATTERN.findall(text)
    if boxed:
        return boxed[-1].strip()
    return None


def _canonical_numeric_token(token: str) -> str | None:
    token = token.strip().replace(" ", "")
    if not token:
        return None

    # Both separators: the right-most one is treated as decimal separator.
    if "," in token and "." in token:
        if token.rfind(",") > token.rfind("."):
            token = token.replace(".", "").replace(",", ".")
        else:
            token = token.replace(",", "")
    elif "," in token:
        left, right = token.rsplit(",", 1)
        # Two decimal digits is overwhelmingly the intended format for euro answers.
        if len(right) in {1, 2}:
            token = left.replace(",", "") + "." + right
        else:
            token = token.replace(",", "")
    elif "." in token:
        # Keep a single decimal dot; multiple dots are thousands + decimal.
        if token.count(".") > 1:
            parts = token.split(".")
            token = "".join(parts[:-1]) + "." + parts[-1]

    try:
        value = Decimal(token)
    except InvalidOperation:
        return None
    if value == value.to_integral():
        return str(int(value))
    normalized = format(value.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def normalize_number(text: str) -> str | None:
    cleaned = (
        text.replace("€", "")
        .replace("EUR", "")
        .replace("eur", "")
        .replace("$", "")
    )
    matches = NUMBER_PATTERN.findall(cleaned)
    if not matches:
        return None
    return _canonical_numeric_token(matches[-1])


def numbers_match(predicted: str | None, expected: str | None) -> bool:
    if predicted is None or expected is None:
        return False
    return normalize_number(predicted) == normalize_number(expected)


def format_reward(prompts: Sequence, completions: Sequence, **kwargs) -> list[float]:
    return [1.0 if has_valid_format(_completion_text(c)) else 0.0 for c in completions]


def accuracy_reward(
    prompts: Sequence, completions: Sequence, answer: Sequence[str], **kwargs
) -> list[float]:
    rewards = []
    for completion, expected in zip(completions, answer, strict=True):
        predicted = extract_answer(_completion_text(completion))
        rewards.append(1.0 if numbers_match(predicted, expected) else 0.0)
    return rewards


def thinking_length(text: str) -> int:
    match = FORMAT_PATTERN.match(text)
    if match is None:
        match = re.search(r"<think>(?P<think>.*?)</think>", text, re.DOTALL)
    if match is None:
        return 0
    return len(match.group("think").split())
