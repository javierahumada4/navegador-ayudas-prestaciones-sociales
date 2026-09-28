"""Checks for the evaluation helpers that do not need a model."""

import json
from pathlib import Path

from rlm.evaluate import (
    diagnose_failure,
    load_problems,
    render_markdown,
    score_row,
    select_failures,
    summarize,
)
from rlm.imv_engine import evaluate_imv_case
from rlm.verifier import build_verifier

TEST_DATA = Path(__file__).resolve().parents[1] / "rlm" / "data" / "test.jsonl"
VERIFIER = build_verifier("imv")


def answer(amount: str) -> str:
    return f"<think>razonamiento</think><answer>{amount}</answer>"


def find_problem(predicate) -> dict:
    for problem in load_problems(str(TEST_DATA), None):
        if predicate(evaluate_imv_case(problem["params"])):
            return problem
    raise AssertionError("no test problem matches")


def test_load_problems_keeps_params_and_limits():
    problems = load_problems(str(TEST_DATA), 3)
    assert len(problems) == 3
    assert {"id", "question", "answer", "branches", "params"} <= set(problems[0])


def test_correct_answer_has_no_failure():
    problem = load_problems(str(TEST_DATA), 1)[0]
    row = score_row(problem, answer(problem["answer"]), 50, VERIFIER, 4096)
    assert row["is_correct"] and row["failure"] is None


def test_missed_capi_is_diagnosed():
    problem = find_problem(lambda o: o["imv_monthly_eur"] > 0 and o["capi_monthly_eur"] > 0)
    imv_only = f"{evaluate_imv_case(problem['params'])['imv_monthly_eur']:.2f}"
    row = score_row(problem, answer(imv_only), 50, VERIFIER, 4096)
    assert row["failure"] == "missed_capi"


def test_zero_answers_are_diagnosed_both_ways():
    eligible = find_problem(lambda o: o["total_monthly_eur"] > 0)
    assert score_row(eligible, answer("0.00"), 50, VERIFIER, 4096)["failure"] == "wrongly_denied"
    ineligible = find_problem(lambda o: o["total_monthly_eur"] == 0)
    row = score_row(ineligible, answer("300.00"), 50, VERIFIER, 4096)
    assert row["failure"] == "missed_ineligibility"
    assert row["failure_details"]["failed_requirements"]


def test_truncated_and_near_miss():
    problem = find_problem(lambda o: o["total_monthly_eur"] > 10)
    cut = score_row(problem, "<think>sin terminar", 4096, VERIFIER, 4096)
    assert cut["truncated"] and cut["failure"] == "truncated"
    near = f"{float(problem['answer']) + 0.5:.2f}"
    assert (
        diagnose_failure(score_row(problem, answer(near), 50, VERIFIER, 4096), problem["params"])[0]
        == "near_miss"
    )


def test_summary_selection_and_markdown():
    problems = load_problems(str(TEST_DATA), 20)
    rows = [
        score_row(p, answer(p["answer"] if i % 2 else "0.00"), 100, VERIFIER, 4096)
        for i, p in enumerate(problems)
    ]
    summary = summarize(rows)
    assert summary["n_problems"] == 20
    assert 0 < summary["pass@1"] < 1
    assert summary["zero_answer_rate"] >= 0.5
    assert sum(f["n"] for f in summary["pass@1_by_family"].values()) == 20
    picked = select_failures(rows, 5)
    assert len(picked) == min(5, sum(1 for r in rows if r["failure"]))
    assert all(r["failure"] for r in picked)
    table = render_markdown({"models": {"base": {"summary": summary}}})
    assert "| base |" in table and "wrongly_denied" in table
    json.dumps(rows)  # rows must be serialisable for the report
