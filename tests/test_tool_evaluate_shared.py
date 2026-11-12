"""CPU-only shared Phase-1/tool evaluation contract tests."""

import json
from types import SimpleNamespace

from api.schemas import ToolCallRecord
from tool_use.evaluate_shared import (
    load_shared_test, prompt_for_case, score_response, summarise,
)


def test_shared_loader_keeps_original_questions_but_never_sends_answers(tmp_path):
    path = tmp_path / "test.jsonl"
    case = {
        "question": "Solicitud 2026. ¿Cuánto IMV+CAPI?",
        "answer": "431.26",
        "params": {"economic": {"secret": "should never reach model"}},
        "rule_context": "Normativa IMV 2026",
        "branches": {"family": "ordinary"},
    }
    path.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
    loaded = load_shared_test(path)
    assert len(loaded) == 1
    prompt = prompt_for_case(loaded[0])
    assert prompt == "Normativa IMV 2026\n\nSolicitud 2026. ¿Cuánto IMV+CAPI?"
    assert "should never reach model" not in prompt
    assert "431.26" not in prompt


def test_numeric_score_matches_phase1_verifier_and_preserves_traces():
    case = {"question": "Caso", "answer": "431.26", "branches": {"family": "ordinary"}}
    record = ToolCallRecord(name="calcular_imv", arguments={"case": {}}, ok=True,
                            result={"total_monthly_eur": 431.26})
    response = SimpleNamespace(answer="431.26", tool_calls=[record])
    row = score_response(case, response, ["original model generation"])
    assert row["numeric_correct"]
    assert row["strict_numeric_format"]
    assert row["tool_execution_ok"]
    assert row["correct_with_tool"]
    assert row["raw_generations"] == ["original model generation"]
    summary = summarise([row])
    assert summary["numeric_accuracy"] == 1.0
    assert summary["correct_with_tool_rate"] == 1.0

    bad = score_response(
        case, SimpleNamespace(answer="IMV 373.76 + CAPI 57.50 = 431.26",
                              tool_calls=[record]), []
    )
    assert not bad["numeric_correct"]  # No reward for multiple amounts.

    no_call = score_response(
        case, SimpleNamespace(answer="431.26", tool_calls=[]), []
    )
    assert no_call["numeric_correct"]
    assert not no_call["tool_execution_ok"]
    assert not no_call["correct_with_tool"]


def test_shared_eval_rejects_invalid_question_or_missing_answer(tmp_path):
    import pytest

    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps({"question": 5, "answer": "1.00"}) + "\n",
                    encoding="utf-8")
    with pytest.raises(ValueError):
        load_shared_test(path)
