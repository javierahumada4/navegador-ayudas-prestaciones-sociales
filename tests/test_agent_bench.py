"""Phase 4 bench and its automatic checks (CPU)."""

import json
from pathlib import Path

import pytest

from agent.bench.build_tasks import tasks as build_tasks
from agent.evaluate import check_answer, parse_number, summarize

BENCH = Path("agent/bench/tasks.jsonl")


def load():
    return [json.loads(line) for line in BENCH.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize("token,value", [
    ("733,60", 733.6), ("733.6", 733.6), ("26.409,60", 26409.6), ("8.803,20", 8803.2),
    ("1,234.5", 1234.5), ("26.409", 26409.0), ("1008.84", 1008.84), ("23", 23.0), ("0,00", 0.0),
])
def test_spanish_and_english_numbers(token, value):
    assert parse_number(token) == pytest.approx(value)


def test_bench_has_twenty_chained_tasks_and_three_impossible():
    rows = load()
    chained = [r for r in rows if r["id"].startswith("T")]
    assert len(chained) == 20 and len([r for r in rows if r["id"].startswith("I")]) == 3
    assert all(len(set(r["required_tools"])) >= 2 for r in chained)
    assert len({r["id"] for r in rows}) == len(rows)


def test_bench_file_matches_the_engine():
    assert load() == json.loads(json.dumps(build_tasks()))


def test_checks_accept_the_right_answer_and_reject_an_invented_one():
    rows = {r["id"]: r for r in load()}
    assert check_answer(rows["T01"], "La edad mínima es 23 años y el total es 733,60 €/mes.")
    assert not check_answer(rows["T01"], "La edad mínima es 23 años y el total es 700 €/mes.")
    assert check_answer(rows["T05"], "Hay 30 días. No tiene derecho al IMV por sus ingresos.")
    assert check_answer(rows["T07"], "El límite es 26.409,60 €; total 0,00 €/mes.")
    assert check_answer(rows["I02"], "No consta en la base de conocimiento.")
    assert not check_answer(rows["I02"], "Cobra 600 euros al mes.")


def test_summary_separates_chained_and_impossible():
    rows = [
        {"id": "T01", "category": "kb+calc", "correct": True, "required_tools_used": True,
         "n_steps": 3, "tokens_used": 100, "wall_s": 10},
        {"id": "T02", "category": "boe+calc", "correct": False, "required_tools_used": False,
         "n_steps": 8, "tokens_used": 300, "wall_s": 30},
        {"id": "I01", "category": "impossible", "correct": True, "required_tools_used": True,
         "n_steps": 1, "tokens_used": 50, "wall_s": 5},
    ]
    s = summarize(rows)
    assert s["success_rate"] == 0.5 and s["impossible_ok"] == 1.0 and s["required_tools_rate"] == 0.5
    assert s["mean_steps"] == 4.0


def test_break_tool_makes_the_tool_fail_like_a_dead_service():
    from agent.evaluate import break_tool
    from agent.react_agent import agent_registry

    registry = break_tool(agent_registry(), "consultar_ayudas")
    record = registry.call("consultar_ayudas", {"tema": "alquiler", "max_resultados": 3})
    assert not record.ok and "ConnectionError" in record.error
    assert agent_registry().tools["consultar_ayudas"].fn is not registry.tools["consultar_ayudas"].fn
