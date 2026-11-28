"""Regression tests for completed reports and phase-2 end-to-end scoring.

Offline, with deterministic scripted model completions. No GPU or BDNS network.
"""
import json
from pathlib import Path

from tool_use.compare import evaluate_configuration
from tool_use.executor import ToolLoop, _money_is_mentioned
from tool_use.evaluate import load_bench
from tool_use.metrics import (
    expected_engine_result, explanation_checks, score_workflow,
)
from tool_use.tools import default_registry


def case_sample(income=3000):
    return {
        "application_date": "2026-06-01",
        "applicant_id": "adult",
        "persons": [{
            "id": "adult", "date_of_birth": "1980-01-01",
            "legal_residence_in_spain": True, "effective_residence_in_spain": True,
            "continuous_legal_effective_residence_since": "2020-01-01",
        }],
        "domicile": {"co_resident_ids": ["adult"], "same_domicile_since": "2020-01-01"},
        "applicant_history": {"independent_from_parents_since": "2010-01-01"},
        "economic": {
            "countable_income_annual_eur": income, "net_worth_eur": 0,
            "non_corporate_assets_eur": 0,
        },
    }


def call(name, args):
    return "<tool_call>" + json.dumps(
        {"name": name, "arguments": args}, ensure_ascii=False,
    ) + "</tool_call>"


def script(*messages):
    loop = ToolLoop()
    outputs = iter(messages)
    loop.generate = lambda _: next(outputs)
    return loop


def model_report(case):
    result = default_registry().call("calcular_imv", {"case": case})
    assert result.ok, result.error
    r = result.result
    return (
        "Informe de simulación orientativa del IMV y CAPI. "
        "Esta evaluación no es una resolución administrativa. "
        f"IMV: {r['imv_monthly_eur']:.2f} euros. "
        f"CAPI: {r['capi_monthly_eur']:.2f} euros. "
        f"Total mensual: {r['total_monthly_eur']:.2f} euros. "
        "Se utilizan los datos normalizados suministrados y las reglas vigentes."
    )


def test_unfinished_report_triggers_second_generation_and_saves(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    facts = case_sample()
    prose = model_report(facts)
    loop = script(
        call("calcular_imv", {"case": facts}),
        "Puedo redactar un informe, pero no lo he guardado todavía.",
        call("guardar_informe", {"contenido": prose}),
    )
    response = loop.run("Calcula IMV y guarda un informe txt.", max_turns=4)
    assert [r.name for r in response.tool_calls] == ["calcular_imv", "guardar_informe"]
    assert all(r.ok for r in response.tool_calls)
    assert response.turns == 3
    assert "guardado correctamente" in response.answer
    assert "informe-imv-" in response.answer
    assert next(tmp_path.glob("*.txt")).read_text(encoding="utf-8") == prose


def test_plain_report_fallback_is_model_written_no_template(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    facts = case_sample()
    prose = model_report(facts)
    result = script(
        call("calcular_imv", {"case": facts}),
        "Tengo los importes pero voy a preparar el informe.",
        prose,
    ).run("Genera y guarda un informe TXT.", max_turns=3)
    assert [x.ok for x in result.tool_calls] == [True, True]
    assert result.tool_calls[-1].name == "guardar_informe"
    assert next(tmp_path.glob("*.txt")).read_text(encoding="utf-8") == prose


def test_false_saved_claim_is_replaced_by_explicit_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    facts = case_sample()
    result = script(
        call("calcular_imv", {"case": facts}),
        "Ya he guardado el documento.",
        "Informe guardado correctamente, ya está preparado.",
    ).run("Calcula IMV y guarda un informe TXT.", max_turns=3)
    assert "No se ha guardado ningún informe" in result.answer
    assert not any(x.ok and x.name == "guardar_informe" for x in result.tool_calls)
    assert not list(tmp_path.glob("*.txt"))


def test_report_max_turns_exhaustion_is_an_explicit_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    result = script(call("calcular_imv", {"case": case_sample()})).run(
        "Genera y guarda un informe TXT.", max_turns=1
    )
    assert "No se ha guardado" in result.answer
    assert "límite" in result.answer
    assert not list(tmp_path.glob("*.txt"))


def test_report_first_requires_observed_calculation(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    facts = case_sample()
    prose = model_report(facts)
    initial = call("calcular_imv", {"case": facts}) + call(
        "guardar_informe", {"contenido": prose}
    )
    result = script(initial, call("guardar_informe", {"contenido": prose})).run(
        "Calcula IMV y genera informe TXT.", max_turns=3
    )
    assert [x.ok for x in result.tool_calls] == [True, False, True]
    assert "earlier turn" in result.tool_calls[1].error
    assert len(list(tmp_path.glob("*.txt"))) == 1


def test_data_missing_never_fabricates_report_or_elides_clarification(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    result = script(
        "Para calcularlo necesito ingresos, patrimonio y fecha de residencia."
    ).run("Quiero un informe TXT de mi IMV, pero aún no tengo datos.", max_turns=3)
    assert result.tool_calls == []
    assert "No se ha guardado" in result.answer
    assert "ingresos" in result.answer and "patrimonio" in result.answer
    assert not list(tmp_path.iterdir())


def test_explicit_request_required_even_when_model_tries_write(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    facts = case_sample()
    result = script(
        call("calcular_imv", {"case": facts}),
        call("guardar_informe", {"contenido": model_report(facts)}),
        "No lo he guardado.",
    ).run("Calcula el IMV sin crear ningún documento.", max_turns=3)
    assert result.tool_calls[0].ok
    assert not result.tool_calls[1].ok
    assert not list(tmp_path.iterdir())


def test_number_parser_accepts_comma_decimals_and_thousands():
    assert _money_is_mentioned("Importe 1.171,74 €", 1171.74)
    assert _money_is_mentioned("importe 1171.7 euros", 1171.70)
    assert _money_is_mentioned("el IMV es 733,60 euros", 733.6)
    assert not _money_is_mentioned("el IMV es 733,60 euros", 732.6)


def test_new_benchmark_has_hard_cases_and_correct_negative_gold():
    cases = load_bench("tool_use/bench/cases.jsonl")
    assert len(cases) == 40
    categories = {c["category"] for c in cases}
    assert categories == {
        "structured_calculation", "natural_calculation", "negative_eligibility",
        "missing_facts", "bdns_natural", "multi_step", "no_tool",
    }
    assert sum(c["category"] == "multi_step" for c in cases) >= 5
    assert sum(c["category"] == "missing_facts" for c in cases) >= 4
    for record in cases:
        if record.get("expected_eligibility"):
            gold = expected_engine_result(record)
            assert gold and gold["status"] == record["expected_eligibility"], record["id"]


def test_full_scoring_verifies_real_report_and_amounts(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    facts = case_sample()
    prose = model_report(facts)
    response = script(
        call("calcular_imv", {"case": facts}),
        call("guardar_informe", {"contenido": prose}),
    ).run("Genera y guarda informe TXT.", max_turns=3)
    calls = [x.model_dump(mode="json") for x in response.tool_calls]
    bench = {
        "needs_tool": True, "expected_tool": "calcular_imv",
        "expected_arguments": {"case": facts},
        "expected_tools": ["calcular_imv", "guardar_informe"],
    }
    score = score_workflow(bench, response.answer, calls, report_root=tmp_path)
    assert score["tool_sequence_success"]
    assert score["multi_step_success"]
    assert score["report_created_and_verified"]
    assert score["amount_fidelity"]
    assert score["explanation_heuristic_accuracy"]
    assert score["end_to_end_success"]

    path = Path(calls[-1]["result"]["archivo"])
    path.write_text("TAMPERED", encoding="utf-8")
    score_after = score_workflow(bench, response.answer, calls, report_root=tmp_path)
    assert not score_after["report_created_and_verified"]


def test_multi_step_requires_all_calls_and_duplicates():
    both = {"needs_tool": True, "expected_tool": "consultar_ayudas",
            "expected_tools": ["consultar_ayudas", "consultar_ayudas"]}
    once = [{"name": "consultar_ayudas", "ok": True}]
    assert not score_workflow(both, "Encontré ayudas", once)["multi_step_success"]
    twice = once + [{"name": "consultar_ayudas", "ok": True}]
    assert score_workflow(both, "Encontré ayudas", twice)["multi_step_success"]


def test_clarification_and_unverified_save_claim_are_scored():
    missing = {
        "needs_tool": False, "expected_behavior": "ask_clarification",
        "expected_clarifications": ["ingres", "patrimon"],
    }
    passing = score_workflow(
        missing, "Necesito conocer tus ingresos y patrimonio.", []
    )
    assert passing["clarification_accuracy"] and passing["end_to_end_success"]
    incorrect = score_workflow(
        {"needs_tool": False}, "He guardado correctamente el informe en disco.", []
    )
    assert incorrect["unverified_save_claim"]
    assert not incorrect["end_to_end_success"]


def test_reason_coverage_checks_denial_codes():
    gold = expected_engine_result({
        "expected_arguments": {"case": {
            **case_sample(), "economic": {
                "countable_income_annual_eur": 40000,
                "net_worth_eur": 0, "non_corporate_assets_eur": 0,
            }
        }}
    })
    assert gold["status"] == "ineligible"
    ok, covered = explanation_checks(
        "Simulación orientativa IMV y CAPI. "
        "Se excluye el IMV debido a los ingresos anuales muy altos.", gold,
    )
    assert ok and covered == 1.0
    bad, covered = explanation_checks("El IMV y el CAPI están aprobados.", gold)
    assert not bad and covered == 0.0


def test_compare_runs_without_gpu_with_scripted_model(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    loop = script("¿Puedes indicar tus ingresos y patrimonio?")
    bench = [{
        "id": "test", "category": "missing_facts", "needs_tool": False,
        "expected_behavior": "ask_clarification",
        "expected_clarifications": ["ingres", "patrimon"],
        "query": "Quiero calcularlo pero me faltan datos.",
    }]
    outcome = evaluate_configuration(bench, loop, max_turns=3, report_root=tmp_path)
    assert outcome["summary"]["n_cases"] == 1
    assert outcome["summary"]["end_to_end_success"] == 1.0
    assert outcome["summary"]["clarification_accuracy_n"] == 1
    assert outcome["rows"][0]["workflow"]["clarification_accuracy"]


def test_report_request_uses_positive_intent_only():
    from tool_use.executor import _report_requested

    assert _report_requested("Genera y guarda un informe TXT de mis resultados.")
    assert _report_requested("Quiero un informe del IMV y CAPI.")
    assert not _report_requested("No quiero un informe por ahora.")
    assert not _report_requested("Calcula el IMV sin crear un documento.")
    assert not _report_requested("¿Cómo generar un informe TXT?")


def test_conceptual_fact_gold_exposes_wrong_acronyms():
    case = {
        "needs_tool": False,
        "expected_facts": ["Ingreso Mínimo Vital"],
    }
    bad = score_workflow(case, "IMV significa Ingresos Mínimos Válidos.", [])
    assert bad["conceptual_fact_accuracy"] is False
    assert not bad["end_to_end_success"]
    correct = score_workflow(case, "IMV significa Ingreso Mínimo Vital.", [])
    assert correct["conceptual_fact_accuracy"] is True
    assert correct["end_to_end_success"]
