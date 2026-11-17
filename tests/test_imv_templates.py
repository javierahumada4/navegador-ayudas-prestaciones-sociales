"""CPU-only checks: deterministic explanation and automatic two-tool chain."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from tool_use.executor import ToolLoop
from tool_use.imv_templates import IMVCalculation, explicar_imv, render_imv_result
from tool_use.tools import default_registry


@pytest.fixture
def case():
    def person(pid, birth, residence):
        return {
            "id": pid,
            "date_of_birth": birth,
            "legal_residence_in_spain": True,
            "effective_residence_in_spain": True,
            "continuous_legal_effective_residence_since": residence,
        }

    return {
        "application_date": "2026-09-26",
        "applicant_id": "p1",
        "persons": [
            person("p1", "1985-05-10", "2010-01-01"),
            person("p2", "1984-01-01", "2010-01-01"),
            person("p3", "2010-06-01", "2010-06-01"),
        ],
        "domicile": {
            "co_resident_ids": ["p1", "p2", "p3"],
            "same_domicile_since": "2010-06-01",
        },
        "relationships": [
            {"person_a": "p1", "person_b": "p2", "type": "spouse", "degree": None},
            {"person_a": "p1", "person_b": "p3", "type": "parent_child", "custody": "shared"},
            {"person_a": "p2", "person_b": "p3", "type": "parent_child", "custody": "shared"},
        ],
        "economic": {
            "countable_income_annual_eur": 9600,
            "net_worth_eur": 5000,
            "non_corporate_assets_eur": 5000,
        },
    }


def test_only_two_imv_tools_exposed_and_teacher_examples_kept():
    from tool_use.tools import calculator, get_weather

    assert callable(calculator.fn) and callable(get_weather.fn)
    registry = default_registry()
    assert set(registry.tools) == {"calcular_imv", "explicar_imv"}
    assert registry.tools["explicar_imv"].requires_confirmation
    schema = registry.tools["explicar_imv"].definition["function"]["parameters"]
    assert schema["required"] == ["resultado"]
    assert "IMVCalculation" in str(schema)


def test_template_saves_real_engine_result_without_llm(case, tmp_path, monkeypatch):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    registry = default_registry()
    calculation = registry.call("calcular_imv", {"case": case})
    assert calculation.ok, calculation.error
    assert calculation.result["total_monthly_eur"] == 431.26

    saved = registry.call("explicar_imv", {"resultado": calculation.result})
    assert saved.ok, saved.error
    report = saved.result
    text = report["texto"]
    assert "431,26 €" in text
    assert "373,76 € de IMV" in text
    assert "57,50 € de CAPI" in text
    assert "3 personas" in text
    assert "no constituye una resolución" in text
    assert text == render_imv_result(IMVCalculation.model_validate(calculation.result))

    path = Path(report["archivo"])
    assert path.is_file()
    assert path.parent == tmp_path
    body = path.read_text(encoding="utf-8")
    assert body == "# Informe orientativo de IMV y CAPI\n\n" + text + "\n"
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == report["sha256"]
    assert path.stat().st_mode & 0o077 == 0

    # A second authorized call creates a second file, without touching the first.
    again = registry.call("explicar_imv", {"resultado": calculation.result})
    assert again.ok, again.error
    assert again.result["texto"] == text
    assert again.result["archivo"] != report["archivo"]
    assert path.read_text(encoding="utf-8") == body
    assert len(list(tmp_path.glob("*.md"))) == 2


def test_capi_only_and_ineligible_are_saved_deterministically(
    case, tmp_path, monkeypatch
):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    registry = default_registry()
    case["economic"]["countable_income_annual_eur"] = 18000
    capi = registry.call("calcular_imv", {"case": case})
    assert capi.ok, capi.error
    assert capi.result["total_monthly_eur"] == 57.50
    report = registry.call("explicar_imv", {"resultado": capi.result})
    assert report.ok, report.error
    assert "0,00 € de IMV" in report.result["texto"]
    assert "57,50 € de CAPI" in report.result["texto"]
    assert "IMV no aplicable" in report.result["texto"]

    case["persons"][0]["active_company_director"] = True
    none = registry.call("calcular_imv", {"case": case})
    assert none.ok, none.error
    assert none.result["status"] == "ineligible"
    report = registry.call("explicar_imv", {"resultado": none.result})
    assert report.ok, report.error
    assert "no reconoce prestación" in report.result["texto"]
    assert "0,00 €" in report.result["texto"]
    assert "administración activa de una sociedad" in report.result["texto"]
    assert len(list(tmp_path.glob("*.md"))) == 2


def test_invalid_engine_output_cannot_create_files(case, tmp_path, monkeypatch):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    registry = default_registry()
    original = registry.call("calcular_imv", {"case": case}).result
    wrong_total = deepcopy(original)
    wrong_total["total_monthly_eur"] = 9999
    result = registry.call("explicar_imv", {"resultado": wrong_total})
    assert not result.ok and "total" in result.error

    wrong_flags = deepcopy(original)
    wrong_flags["imv_eligible"] = False
    result = registry.call("explicar_imv", {"resultado": wrong_flags})
    assert not result.ok and "IMV" in result.error

    wrong_field = deepcopy(original)
    wrong_field["made_up"] = True
    assert not registry.call("explicar_imv", {"resultado": wrong_field}).ok
    with pytest.raises(Exception):
        IMVCalculation.model_validate(wrong_field)
    assert list(tmp_path.iterdir()) == []


class ScriptedImvLoop(ToolLoop):
    def __init__(self, completions):
        super().__init__(registry=default_registry(), brain="base", model_id="fake")
        self._completions = iter(completions)
        self.messages_seen = []

    def generate(self, messages):
        self.messages_seen.append(deepcopy(messages))
        return next(self._completions)


def _calculate_call(case):
    return "<tool_call>" + json.dumps({
        "name": "calcular_imv", "arguments": {"case": case},
    }) + "</tool_call>"


def test_without_confirmation_renders_template_without_writing(
    case, tmp_path, monkeypatch
):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    query = _calculate_call(case)
    loop = ScriptedImvLoop([query])
    trace = []
    result = loop.run("Calcula y explica el IMV", max_turns=3, trace=trace)
    assert len(loop.messages_seen) == 1
    assert [r.name for r in result.tool_calls] == ["calcular_imv"]
    assert all(r.ok for r in result.tool_calls)
    assert result.turns == 1
    assert "431,26 €" in result.answer
    assert "Informe guardado" not in result.answer
    assert trace == [query]
    assert list(tmp_path.iterdir()) == []


def test_confirmation_writes_report_and_records_both_tools(
    case, tmp_path, monkeypatch
):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    loop = ScriptedImvLoop([_calculate_call(case)])
    result = loop.run("Calcula y guarda el informe", confirmar_guardado=True)
    assert len(loop.messages_seen) == 1  # No model-written final response.
    assert [r.name for r in result.tool_calls] == ["calcular_imv", "explicar_imv"]
    assert all(r.ok for r in result.tool_calls)
    assert result.turns == 1
    file = Path(result.tool_calls[1].result["archivo"])
    assert file.is_file()
    assert "431,26 €" in file.read_text(encoding="utf-8")
    assert str(file) in result.answer
    assert result.tool_calls[1].result["texto"] in result.answer


def test_model_cannot_forge_direct_report_write_even_with_confirmation(
    case, tmp_path, monkeypatch
):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    calculation = default_registry().call("calcular_imv", {"case": case})
    call = "<tool_call>" + json.dumps({
        "name": "explicar_imv", "arguments": {"resultado": calculation.result},
    }) + "</tool_call>"
    loop = ScriptedImvLoop([call, "No es posible escribir directamente."])
    result = loop.run("Escribe el informe", confirmar_guardado=True)
    assert result.tool_calls[0].name == "explicar_imv"
    assert not result.tool_calls[0].ok
    assert "application-managed" in result.tool_calls[0].error
    assert list(tmp_path.iterdir()) == []


def test_conceptual_query_does_not_trigger_tools():
    loop = ScriptedImvLoop(["El IMV es una prestación de la Seguridad Social."])
    result = loop.run("¿Qué es el IMV?")
    assert result.tool_calls == []
    assert result.turns == 1


def test_phase2_scoring_still_uses_expected_first_call(case):
    from tool_use.evaluate import score_case

    call = "<tool_call>" + json.dumps({
        "name": "calcular_imv", "arguments": {"case": case},
    }) + "</tool_call>"
    response = ScriptedImvLoop([call]).run("Calcular")
    expected = {
        "needs_tool": True,
        "expected_tool": "calcular_imv",
        "expected_arguments": {"case": case},
    }
    outcome = score_case(expected, [r.model_dump() for r in response.tool_calls])
    assert outcome == {"selection_ok": True, "arguments_ok": True}
