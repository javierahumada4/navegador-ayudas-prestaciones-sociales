"""Phase-2 tests: registry contracts, HTTP failures, file effects and manual loop.

All tests run offline without a GPU; Qwen generation is replaced by scripted responses.
"""
import json
from datetime import date

import httpx
import pytest

from tool_use.domain_tools import calcular_imv, consultar_ayudas, guardar_informe, rules_for_date
from tool_use.executor import ToolLoop
from tool_use.tools import default_registry


def example_case():
    return {
        "application_date": "2026-06-01",
        "applicant_id": "a",
        "persons": [
            {
                "id": "a", "date_of_birth": "1980-01-01",
                "legal_residence_in_spain": True,
                "effective_residence_in_spain": True,
                "continuous_legal_effective_residence_since": "2020-01-01",
            },
            {
                "id": "b", "date_of_birth": "2018-01-01",
                "legal_residence_in_spain": True,
                "effective_residence_in_spain": True,
                "continuous_legal_effective_residence_since": "2020-01-01",
            },
        ],
        "domicile": {"co_resident_ids": ["a", "b"], "same_domicile_since": "2020-01-01"},
        "relationships": [
            {"person_a": "a", "person_b": "b", "type": "parent_child", "custody": "exclusive"}
        ],
        "economic": {
            "countable_income_annual_eur": 6000,
            "net_worth_eur": 0, "non_corporate_assets_eur": 0,
        },
    }


def tool_block(name, arguments):
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"


def scripted_loop(*completions, registry=None):
    loop = ToolLoop(registry=registry or default_registry())
    remaining = iter(completions)
    loop.generate = lambda _: next(remaining)
    return loop


def test_registry_exposes_exactly_three_domain_tools():
    registry = default_registry()
    assert set(registry.tools) == {"calcular_imv", "consultar_ayudas", "guardar_informe"}
    assert all(x["function"]["parameters"] for x in registry.definitions())
    assert registry.tools["guardar_informe"].requires_confirmation


def test_calculator_uses_engine_and_ruleset():
    record = default_registry().call("calcular_imv", {"case": example_case()})
    assert record.ok, record.error
    assert record.result["ruleset_id"] == "imv_es_2026"
    assert record.result["total_monthly_eur"] == (
        record.result["imv_monthly_eur"] + record.result["capi_monthly_eur"]
    )


def test_calculator_rejects_invalid_or_missing_facts():
    case = example_case()
    del case["economic"]["net_worth_eur"]
    record = default_registry().call("calcular_imv", {"case": case})
    assert not record.ok and "invalid arguments" in record.error
    with pytest.raises(ValueError, match="ruleset"):
        rules_for_date(date(2025, 1, 1))


def test_bdns_request_and_structured_response(monkeypatch):
    def fake_get(url, *, params, headers, timeout, follow_redirects):
        assert url.startswith("https://www.infosubvenciones.es/")
        assert params["descripcion"] == "alquiler"
        assert params["pageSize"] == 3
        return httpx.Response(
            200, json={"content": [
                {
                    "numeroConvocatoria": 987654, "descripcion": "Subvenciones de alquiler",
                    "fechaRecepcion": "2026-04-01", "nivel1": "AUTONÓMICA",
                    "nivel2": "Comunidad de ejemplo",
                }
            ], "totalElements": 1},
            request=httpx.Request("GET", url),
        )
    monkeypatch.setattr(httpx, "get", fake_get)
    result = default_registry().call("consultar_ayudas", {"tema": "alquiler", "max_resultados": 3})
    assert result.ok, result.error
    assert result.result["resultados"][0]["codigo_bdns"] == "987654"
    assert "no implica" in result.result["advertencia"]


def test_bdns_error_is_observation(monkeypatch):
    def unavailable(*args, **kwargs):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(httpx, "get", unavailable)
    result = default_registry().call("consultar_ayudas", {"tema": "familias"})
    assert not result.ok and "BDNS unavailable" in result.error


def test_report_file_contains_model_text_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    contenido = "Informe orientativo generado por el modelo. " * 4
    result = default_registry().call("guardar_informe", {"contenido": contenido})
    assert result.ok, result.error
    from pathlib import Path
    path = Path(result.result["archivo"])
    assert path.parent == tmp_path
    assert path.suffix == ".txt" and path.read_text(encoding="utf-8") == contenido
    assert result.result["sha256"]
    assert len(list(tmp_path.iterdir())) == 1


def test_no_tool_needed():
    response = scripted_loop("El IMV es una prestación no contributiva.").run("¿Qué es el IMV?")
    assert response.tool_calls == [] and response.turns == 1


def test_multistep_calculation_then_answer():
    case = example_case()
    loop = scripted_loop(tool_block("calcular_imv", {"case": case}), "La simulación es orientativa.")
    response = loop.run("Calcula el IMV del JSON proporcionado.")
    assert response.turns == 2 and response.tool_calls[0].ok
    assert response.tool_calls[0].result["ruleset_id"] == "imv_es_2026"


def test_parallel_calls_then_answer(monkeypatch):
    monkeypatch.setattr(
        httpx, "get",
        lambda url, **kwargs: httpx.Response(
            200, json={"content": []}, request=httpx.Request("GET", url)
        ),
    )
    calls = tool_block("calcular_imv", {"case": example_case()}) + tool_block(
        "consultar_ayudas", {"tema": "alquiler"}
    )
    response = scripted_loop(calls, "He revisado ambos resultados.").run("Calcula y busca alquiler.")
    assert response.turns == 2 and len(response.tool_calls) == 2
    assert all(record.ok for record in response.tool_calls)


def test_invalid_arguments_observed_and_corrected():
    loop = scripted_loop(
        tool_block("calcular_imv", {"case": {"application_date": "2026-01-01"}}),
        tool_block("calcular_imv", {"case": example_case()}),
        "He calculado la simulación.",
    )
    result = loop.run("Calcula IMV.")
    assert len(result.tool_calls) == 2
    assert not result.tool_calls[0].ok and result.tool_calls[1].ok


def test_report_requires_explicit_request_and_prior_engine(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    report = "Simulación orientativa de IMV/CAPI. " * 5
    generated = tool_block("guardar_informe", {"contenido": report})
    result = scripted_loop(generated, "No he escrito nada.").run("Dime el resultado.")
    assert not result.tool_calls[0].ok
    assert not list(tmp_path.iterdir())
    result = scripted_loop(generated, "Falta el cálculo.").run("Genera el informe txt.")
    assert not result.tool_calls[0].ok
    assert not list(tmp_path.iterdir())


def test_report_generated_by_model_after_calculation(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    case = example_case()
    result = default_registry().call("calcular_imv", {"case": case})
    assert result.ok
    parts = result.result
    report = (
        "Informe orientativo de IMV/CAPI basado en una simulación; no es "
        "una resolución de la Seguridad Social. "
        f"IMV: {parts['imv_monthly_eur']:.2f} euros. "
        f"CAPI: {parts['capi_monthly_eur']:.2f} euros. "
        f"Total mensual: {parts['total_monthly_eur']:.2f} euros."
    )
    loop = scripted_loop(
        tool_block("calcular_imv", {"case": case}),
        tool_block("guardar_informe", {"contenido": report}),
        "Informe guardado.",
    )
    answer = loop.run("Genera un informe txt del IMV/CAPI de este caso.")
    assert [r.ok for r in answer.tool_calls] == [True, True]
    assert len(list(tmp_path.glob("*.txt"))) == 1


def test_incomplete_call_and_duplicate_are_observations():
    incomplete = scripted_loop('<tool_call>{"name":"calcular_imv"', "No pude ejecutar.")
    out = incomplete.run("Calcula")
    assert not out.tool_calls[0].ok
    call = tool_block("calcular_imv", {"case": example_case()})
    result = scripted_loop(call, call, "Fin").run("Calcula")
    assert result.tool_calls[0].ok and not result.tool_calls[1].ok
