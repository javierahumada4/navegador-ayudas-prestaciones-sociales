"""Teacher traces are rebuilt into exactly one <think> and one <answer> block."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("datasets")

from rlm.distill import _structure_trace  # noqa: E402
from rlm.rewards import has_valid_format  # noqa: E402

CANONICAL = "<think>\nRazonamiento.\n</think>\n<answer>474.01</answer>"


def test_clean_trace_is_kept():
    trace, status = _structure_trace("<think>\nRazonamiento.\n</think>\n\n<answer>474.01</answer>")
    assert (trace, status) == (CANONICAL, "clean")


def test_duplicated_tags_from_pilot_are_repaired():
    raw = (
        "<think>\n<think>\nRazonamiento.\n</think>\n\n</think>\n\n"
        "<answer>474.01</answer></think><answer>474.01</answer>"
    )
    assert _structure_trace(raw) == (CANONICAL, "repaired")


def test_different_amounts_are_rejected():
    raw = "<think>\nR\n</think>\n<answer>474.01</answer> o <answer>0.00</answer>"
    assert _structure_trace(raw) == (None, "ambiguous_answer")


def test_truncated_thinking_is_rejected():
    assert _structure_trace("<think>\nR que no acaba") == (None, "unclosed_think")


def test_missing_answer_is_rejected():
    assert _structure_trace("<think>\nR\n</think>\nEl total es 474,01 €") == (None, "no_answer")


def test_empty_reasoning_is_rejected():
    assert _structure_trace("<think>\n</think>\n<answer>1.00</answer>") == (None, "empty_reasoning")


def test_spanish_amount_and_zero_are_normalized():
    assert _structure_trace("<think>R</think><answer>474,01 €</answer>")[0].endswith(
        "<answer>474.01</answer>"
    )
    assert _structure_trace("<think>R</think><answer>0</answer>")[0].endswith(
        "<answer>0.00</answer>"
    )


def test_answer_tags_inside_reasoning_are_removed():
    raw = "<think>Formato: <answer>1.00</answer>. Total 474.01</think><answer>474.01</answer>"
    trace, status = _structure_trace(raw)
    assert status == "repaired"
    assert trace.count("<answer>") == 1 and has_valid_format(trace)
