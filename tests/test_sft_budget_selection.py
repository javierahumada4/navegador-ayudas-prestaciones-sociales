"""SFT selects complete teacher traces using the student's completion tokens."""
import json

import pytest

pytest.importorskip("datasets")
from datasets import Dataset  # noqa: E402

from rlm.sft_selection import select_budgeted_traces  # noqa: E402


class ByteTokenizer:
    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=True,
    ):
        assert enable_thinking
        prefix = "<user>REGLAS</user><assistant>"
        if add_generation_prompt:
            return prefix
        return prefix + messages[-1]["content"] + "</assistant>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text.encode("utf-8"))}


def row(question, result, *, family="one", verified=True):
    return {"question": question, "trace": result, "verified": verified,
            "rule_context": "REGLAS", "branches": {"family": family}}


def trace(thought):
    return f"<think>{thought}</think><answer>57.50</answer>"


def test_select_short_verified_traces_and_report_missing_coverage(tmp_path):
    short_a = trace("Correcto.")
    short_b = trace("Otra forma.")
    long = trace("x" * 150)
    raw = Dataset.from_list([
        row("A", long), row("A", short_a), row("A", short_b),
        row("A", short_a), row("B", long, family="two"),
        row("C", short_a, family="two", verified=False),
    ])
    path = tmp_path / "selection.json"
    selected, report = select_budgeted_traces(
        raw, ByteTokenizer(), max_completion_tokens=90, max_length=600,
        max_per_problem=2, min_problem_coverage=0.30, min_families=1,
        report_path=path,
    )
    assert list(selected["question"]) == ["A", "A"]
    assert set(selected["trace"]) == {short_a, short_b}
    assert report["total_problems"] == 3
    assert report["covered_problems"] == 1
    assert report["excluded_by_reason"] == {"over_completion_budget": 2}
    assert report["duplicate_traces_skipped"] == 1
    assert json.loads(path.read_text())["problem_coverage"] == 0.333333


def test_coverage_gate_fails_with_report_before_training(tmp_path):
    data = Dataset.from_list([
        row("A", trace("Breve.")),
        row("B", trace("x" * 150), family="two"),
    ])
    path = tmp_path / "coverage.json"
    with pytest.raises(ValueError, match="Insufficient SFT-eligible coverage"):
        select_budgeted_traces(
            data, ByteTokenizer(), max_completion_tokens=90, max_length=600,
            max_per_problem=2, min_problem_coverage=0.70, min_families=2,
            report_path=path,
        )
    saved = json.loads(path.read_text())
    assert saved["traces_selected"] == 1
    assert saved["covered_families"] == 1


def test_context_length_gate_still_applies(tmp_path):
    data = Dataset.from_list([row("A", trace("x" * 70))])
    path = tmp_path / "context.json"
    with pytest.raises(ValueError, match="Insufficient SFT-eligible coverage"):
        select_budgeted_traces(
            data, ByteTokenizer(), max_completion_tokens=130, max_length=135,
            max_per_problem=1, min_problem_coverage=1.0, min_families=1,
            report_path=path,
        )
    assert json.loads(path.read_text())["excluded_by_reason"] == {
        "over_context_budget": 1,
    }


def test_low_coverage_can_warn_and_train_on_available_verified_traces(tmp_path):
    short = trace("Breve.")
    long = trace("x" * 150)
    raw = Dataset.from_list([
        row("A", short), row("B", long, family="two")
    ])
    selected, report = select_budgeted_traces(
        raw, ByteTokenizer(), max_completion_tokens=90, max_length=600,
        max_per_problem=2, min_problem_coverage=0.70, min_families=2,
        report_path=tmp_path / "low_coverage.json", warn_low_coverage=True,
    )
    assert len(selected) == 1
    assert selected[0]["trace"] == short
    assert report["quality_passed"] is False
    assert "Insufficient SFT-eligible coverage" in report["quality_warnings"][0]
    assert json.loads((tmp_path / "low_coverage.json").read_text())["traces_selected"] == 1


def test_warning_mode_never_allows_empty_sft_dataset(tmp_path):
    raw = Dataset.from_list([row("A", trace("x" * 150))])
    with pytest.raises(ValueError, match="No verified complete SFT traces"):
        select_budgeted_traces(
            raw, ByteTokenizer(), max_completion_tokens=90, max_length=600,
            max_per_problem=2, min_problem_coverage=0.70, min_families=1,
            report_path=tmp_path / "empty.json", warn_low_coverage=True,
        )
    assert json.loads((tmp_path / "empty.json").read_text())["traces_selected"] == 0
