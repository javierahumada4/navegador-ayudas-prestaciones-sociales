"""Deterministic verifiers for phase-1 RLVR.

The IMV verifier is deliberately strict about the *value* and tolerant about
presentation (decimal comma/dot, optional euro symbol).  Ground truth comes from
rlm.imv_engine via the generated JSONL files.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from rlm.rewards import extract_answer, normalize_number


@dataclass(frozen=True)
class VerificationResult:
    is_correct: bool
    predicted: str | None
    expected: str
    detail: str = ""


class Verifier(ABC):
    name: str = "verifier"

    @abstractmethod
    def is_correct(self, predicted: str | None, expected: str) -> bool:
        ...

    def verify(self, completion: str, expected: str) -> VerificationResult:
        predicted = extract_answer(completion)
        ok = self.is_correct(predicted, expected)
        detail = "no <answer> block found" if predicted is None else ""
        return VerificationResult(ok, predicted, expected, detail)


class NumericVerifier(Verifier):
    name = "numeric"

    def __init__(self, tolerance: float = 0.0):
        self.tolerance = tolerance

    def is_correct(self, predicted: str | None, expected: str) -> bool:
        if predicted is None:
            return False
        p, e = normalize_number(predicted), normalize_number(expected)
        if p is None or e is None:
            return False
        if self.tolerance == 0.0:
            return p == e
        return abs(float(p) - float(e)) <= self.tolerance


class ExactMatchVerifier(Verifier):
    name = "exact_match"

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def is_correct(self, predicted: str | None, expected: str) -> bool:
        return predicted is not None and self._normalize(predicted) == self._normalize(expected)


class IMVAmountVerifier(Verifier):
    """Compare the final monthly IMV+CAPI amount to the cent."""

    name = "imv"

    @staticmethod
    def _amount(text: str | None) -> Decimal | None:
        if text is None:
            return None
        normalized = normalize_number(text)
        if normalized is None:
            return None
        try:
            return Decimal(normalized).quantize(Decimal("0.01"))
        except InvalidOperation:
            return None

    def is_correct(self, predicted: str | None, expected: str) -> bool:
        p = self._amount(predicted)
        e = self._amount(expected)
        return p is not None and e is not None and p == e


VERIFIER_FACTORIES: dict[str, type[Verifier]] = {
    "numeric": NumericVerifier,
    "exact_match": ExactMatchVerifier,
    "imv": IMVAmountVerifier,
}


def build_verifier(name: str) -> Verifier:
    try:
        return VERIFIER_FACTORIES[name]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown verifier {name!r}; choose one of {sorted(VERIFIER_FACTORIES)}"
        ) from exc
