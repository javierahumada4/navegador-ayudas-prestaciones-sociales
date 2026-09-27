"""Verifiable reward helpers for phase 1.

The generic numeric reward keeps exact numeric semantics. IMV additionally uses
``imv_accuracy_reward`` so training, the verifier and the deterministic oracle
share the same monetary convention: Decimal + ROUND_HALF_UP to cents.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

CENT = Decimal("0.01")

FORMAT_PATTERN = re.compile(
    r"^\s*<think>(?P<think>(?:(?!</?think>).)*)</think>"
    r"\s*<answer>(?P<answer>(?:(?!</?answer>).)*)</answer>\s*$",
    re.DOTALL,
)
ANSWER_PATTERN = re.compile(r"<answer>(?P<answer>.*?)</answer>", re.DOTALL)
BOXED_PATTERN = re.compile(r"\\boxed\{(?P<answer>[^{}]*)\}")
# Permissive token; separator semantics are resolved below.
NUMBER_PATTERN = re.compile(r"-?\d(?:[\d.,\s]*\d)?")


def _completion_text(completion: str | Sequence[dict]) -> str:
    if isinstance(completion, str):
        return completion
    parts = []
    for message in completion:
        if isinstance(message, dict) and message.get("role", "assistant") == "assistant":
            parts.append(str(message.get("content", "")))
    return "\n".join(parts)


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


def _decimal_from_token(token: str) -> Decimal | None:
    """Parse common English/Spanish number formatting without going through float.

    Examples:
      1,234.50 -> 1234.50
      1.173,76 -> 1173.76
      431,26   -> 431.26
      1,234    -> 1234  (legacy repo/GSM8K convention)
    """
    raw = token.replace(" ", "")
    if not raw:
        return None

    sign = ""
    if raw.startswith("-"):
        sign, raw = "-", raw[1:]

    if "," in raw and "." in raw:
        # The right-most separator is decimal; the other is thousands.
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif "," in raw:
        parts = raw.split(",")
        if len(parts) == 2:
            left, right = parts
            # Preserve repository behaviour for e.g. 1,234, but interpret 431,26 as decimal.
            if len(right) == 3 and 1 <= len(left) <= 3:
                raw = left + right
            else:
                raw = left + "." + right
        else:
            # Multiple commas: standard thousands if every tail group has three digits;
            # otherwise use the last comma as decimal separator.
            if all(len(x) == 3 for x in parts[1:]):
                raw = "".join(parts)
            else:
                raw = "".join(parts[:-1]) + "." + parts[-1]
    elif raw.count(".") > 1:
        parts = raw.split(".")
        if all(len(x) == 3 for x in parts[1:]):
            raw = "".join(parts)
        else:
            raw = "".join(parts[:-1]) + "." + parts[-1]

    try:
        return Decimal(sign + raw)
    except InvalidOperation:
        return None


def parse_number_decimal(text: str | None) -> Decimal | None:
    """Extract the last numeric token and return an exact Decimal."""
    if text is None:
        return None
    cleaned = (
        text.replace("€", "")
        .replace("$", "")
        .replace("EUR", "")
        .replace("eur", "")
    )
    numbers = NUMBER_PATTERN.findall(cleaned)
    if not numbers:
        return None
    return _decimal_from_token(numbers[-1])




def parse_imv_decimal(text: str | None) -> Decimal | None:
    """Parse an IMV monetary answer.

    In this Spanish-domain parser a lone comma is a decimal separator, even
    when three digits follow it (e.g. ``474,005``). Mixed ``1.234,56`` and
    ``1,234.56`` formats are both accepted.
    """
    if text is None:
        return None
    cleaned = (
        text.replace("€", "")
        .replace("$", "")
        .replace("EUR", "")
        .replace("eur", "")
    )
    numbers = NUMBER_PATTERN.findall(cleaned)
    if not numbers:
        return None
    raw = numbers[-1].replace(" ", "")
    sign = ""
    if raw.startswith("-"):
        sign, raw = "-", raw[1:]
    if "," in raw and "." in raw:
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif "," in raw:
        if raw.count(",") == 1:
            raw = raw.replace(",", ".")
        else:
            parts = raw.split(",")
            raw = "".join(parts[:-1]) + "." + parts[-1]
    elif raw.count(".") > 1:
        parts = raw.split(".")
        # Spanish thousands groups such as 1.173.760 have no decimal part.
        if all(len(x) == 3 for x in parts[1:]):
            raw = "".join(parts)
        else:
            raw = "".join(parts[:-1]) + "." + parts[-1]
    try:
        return Decimal(sign + raw)
    except InvalidOperation:
        return None


def normalize_number(text: str) -> str | None:
    """Return a canonical decimal string for the last number in *text*."""
    value = parse_number_decimal(text)
    if value is None:
        return None
    if value == 0:
        return "0"
    out = format(value, "f")
    if "." in out:
        out = out.rstrip("0").rstrip(".")
    return out


def numbers_match(predicted: str | None, expected: str | None) -> bool:
    """Generic numeric equality; no domain-specific rounding."""
    if predicted is None or expected is None:
        return False
    p, e = parse_number_decimal(predicted), parse_number_decimal(expected)
    return p is not None and e is not None and p == e


def imv_money(value: Decimal) -> Decimal:
    """Canonical IMV money rounding used by oracle/verifier/reward."""
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def imv_numbers_match(predicted: str | None, expected: str | None) -> bool:
    """Compare monetary answers after HALF_UP quantisation to cents."""
    if predicted is None or expected is None:
        return False
    p, e = parse_imv_decimal(predicted), parse_imv_decimal(expected)
    if p is None or e is None:
        return False
    return imv_money(p) == imv_money(e)


def format_reward(prompts: Sequence, completions: Sequence, **kwargs) -> list[float]:
    return [1.0 if has_valid_format(_completion_text(c)) else 0.0 for c in completions]


def accuracy_reward(
    prompts: Sequence, completions: Sequence, answer: Sequence[str], **kwargs
) -> list[float]:
    """Generic exact numeric reward retained for GSM8K/smoke tests."""
    rewards = []
    for completion, expected in zip(completions, answer, strict=True):
        predicted = extract_answer(_completion_text(completion))
        rewards.append(1.0 if numbers_match(predicted, expected) else 0.0)
    return rewards


def imv_accuracy_reward(
    prompts: Sequence, completions: Sequence, answer: Sequence[str], **kwargs
) -> list[float]:
    """IMV accuracy reward: same HALF_UP-to-cents semantics as the oracle."""
    rewards = []
    for completion, expected in zip(completions, answer, strict=True):
        predicted = extract_answer(_completion_text(completion))
        rewards.append(1.0 if imv_numbers_match(predicted, expected) else 0.0)
    return rewards


def thinking_length(text: str) -> int:
    match = FORMAT_PATTERN.match(text)
    if match is None:
        match = re.search(r"<think>(?P<think>.*?)</think>", text, re.DOTALL)
    if match is None:
        return 0
    return len(match.group("think").split())
