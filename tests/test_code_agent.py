"""Phase 4 contrast: smolagents CodeAgent on our model pool and tools (scripted model, CPU)."""

import json

import pytest

pytest.importorskip("smolagents")

from agent.code_agent import run_code_agent, to_chat  # noqa: E402
from tool_use.domain_tools import calcular_imv  # noqa: E402
from tool_use.registry import ToolRegistry  # noqa: E402

CASE = {
    "application_date": "2026-06-01", "applicant_id": "adult",
    "persons": [{"id": "adult", "date_of_birth": "1980-01-01", "legal_residence_in_spain": True,
                 "effective_residence_in_spain": True,
                 "continuous_legal_effective_residence_since": "2020-01-01"}],
    "domicile": {"co_resident_ids": ["adult"], "same_domicile_since": "2020-01-01"},
    "relationships": [], "applicant_history": {"independent_from_parents_since": "2010-01-01"},
    "economic": {"countable_income_annual_eur": 0, "net_worth_eur": 0, "non_corporate_assets_eur": 0},
}


def scripted(*outputs):
    calls = []

    def generator(messages, spec, tools):
        calls.append(messages)
        return outputs[min(len(calls), len(outputs)) - 1], 100, 20

    generator.calls = calls
    return generator


def code(body: str) -> str:
    return f"Thought: lo calculo con el motor.\n<code>\n{body}\n</code>"


def registry():
    return ToolRegistry().register(calcular_imv)


def test_code_agent_calls_our_tools_and_answers():
    snippet = f"r = calcular_imv(case={CASE!r})\nfinal_answer(r['total_monthly_eur'])"
    out = run_code_agent("¿Cuánto IMV?", max_steps=3, brain="base",
                         generator=scripted(code(snippet)), registry_factory=registry)
    assert out["succeeded"], out
    assert out["final_answer"] == "733.6"
    assert out["tools_used"] == ["calcular_imv"] and out["n_steps"] == 1
    assert out["tokens_used"] == 120


def test_tool_errors_come_back_as_exceptions_and_the_agent_recovers():
    bad = code("r = calcular_imv(case={'applicant_id': 'x'})\nprint(r)")
    good = code(f"r = calcular_imv(case={CASE!r})\nfinal_answer(r['total_monthly_eur'])")
    gen = scripted(bad, good)
    out = run_code_agent("¿Cuánto IMV?", max_steps=4, brain="base", generator=gen,
                         registry_factory=registry)
    assert out["succeeded"] and out["n_steps"] == 2
    assert "invalid arguments" in out["steps"][0]["observation"]
    # the error reached the model in the second prompt
    assert "invalid arguments" in json.dumps(gen.calls[1], ensure_ascii=False)


def test_messages_are_flattened_to_alternating_chat():
    chat = to_chat([
        {"role": "system", "content": [{"type": "text", "text": "s"}]},
        {"role": "user", "content": [{"type": "text", "text": "a"}]},
        {"role": "tool-response", "content": [{"type": "text", "text": "b"}]},
    ])
    assert chat == [{"role": "system", "content": "s"}, {"role": "user", "content": "a\nb"}]
