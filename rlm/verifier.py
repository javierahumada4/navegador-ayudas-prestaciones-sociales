"""Deterministic answer verifiers for phase 1."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal

from rlm.rewards import extract_answer, imv_numbers_match, parse_number_decimal


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
    """Generic numeric comparison. This is intentionally not IMV-specific."""

    name = "numeric"

    def __init__(self, tolerance: float = 0.0):
        self.tolerance = Decimal(str(tolerance))

    def is_correct(self, predicted: str | None, expected: str) -> bool:
        p = parse_number_decimal(predicted)
        e = parse_number_decimal(expected)
        if p is None or e is None:
            return False
        if self.tolerance == 0:
            return p == e
        return abs(p - e) <= self.tolerance


class IMVAmountVerifier(Verifier):
    """IMV/CAPI verifier using the oracle's ROUND_HALF_UP-to-cents convention."""

    name = "imv"

    def is_correct(self, predicted: str | None, expected: str) -> bool:
        return imv_numbers_match(predicted, expected)


class ExactMatchVerifier(Verifier):
    name = "exact_match"

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    def is_correct(self, predicted: str | None, expected: str) -> bool:
        if predicted is None:
            return False
        return self._normalize(predicted) == self._normalize(expected)


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
            f"unknown verifier {name!r}; choose one of {sorted(VERIFIER_FACTORIES)}"
        ) from exc
