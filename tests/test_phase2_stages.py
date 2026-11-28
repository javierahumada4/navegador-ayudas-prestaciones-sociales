"""Stage isolation uses real Pydantic tools and phase-4-shaped tool observations.

Offline tests script only the model's generations, not deterministic IMV execution.
"""
import json
from pathlib import Path

import httpx

from tool_use.evaluate import load_bench
from tool_use.executor import ToolLoop, build_system_prompt
from tool_use.metrics import score_workflow
from tool_use.stage_eval import (
    STAGES, evaluate_stage_cases, registry_for_stage, score_stage,
    seed_report_observations,
)


def block(name, arguments):
    return "<tool_call>" + json.dumps(
        {"name": name, "arguments": arguments}, ensure_ascii=False
    ) + "</tool_call>"


def scripted(stage, *generations):
    loop = ToolLoop(registry=registry_for_stage(stage))
    outputs = iter(generations)
    loop.generate = lambda messages: next(outputs)
    return loop


def sample(stage, predicate=lambda _: True):
    return next(c for c in load_bench(f"tool_use/bench/{stage}.jsonl") if predicate(c))


def good_report(calculation, api, *, cite=True):
    result = calculation.result
    source = ("BDNS: " + str(api.result["resultados"][0]["codigo_bdns"])
              + " es una convocatoria FICTICIA de evaluación."
              if api.ok and api.result["resultados"] and cite
              else "BDNS no se pudo consultar, la búsqueda está no disponible."
              if not api.ok
              else "BDNS: no se encontraron convocatorias en esta búsqueda.")
    return (
        "Informe de simulación orientativa del IMV/CAPI. Este texto no constituye "
        "una resolución de reconocimiento de derechos. "
        f"IMV: {result['imv_monthly_eur']:.2f} euros. "
        f"CAPI: {result['capi_monthly_eur']:.2f} euros. "
        f"Total mensual: {result['total_monthly_eur']:.2f} euros. "
        + ("Los ingresos o renta familiar superan los límites exigidos. "
           if result.get("failed_requirements") else "")
        + source
    )


def test_four_files_have_disjoint_task_categories_and_ground_truth():
    lengths = {"imv": 21, "bdns": 10, "report": 6, "integration": 8}
    assert set(STAGES) == set(lengths)
    for name, size in lengths.items():
        rows = load_bench(f"tool_use/bench/{name}.jsonl")
        assert len(rows) == size
        assert all(c["stage"] == name for c in rows)
        assert len(set(c["id"] for c in rows)) == size


def test_only_relevant_tool_is_shown_for_each_stage():
    for stage, expected in (
        ("imv", {"calcular_imv"}),
        ("bdns", {"consultar_ayudas"}),
        ("report", {"guardar_informe"}),
        ("integration", {"calcular_imv", "consultar_ayudas", "guardar_informe"}),
    ):
        registry = registry_for_stage(stage)
        assert set(registry.tools) == expected
        # Schemas reach the model once, through the chat template's tools=, not the prompt.
        assert "Definiciones disponibles" not in build_system_prompt(registry)
        rendered = registry.render_for_prompt()
        for name in expected:
            assert f'"name": "{name}"' in rendered
    assert "objeto llamado case" in build_system_prompt(registry_for_stage("imv"))


def test_imv_isolated_computes_and_grades_canonical_json():
    case = sample("imv", lambda c: c["category"] == "structured_calculation")
    prompt = case["query"]
    answer = scripted(
        "imv", block("calcular_imv", case["expected_arguments"]),
        "Simulación orientativa IMV y CAPI: IMV 733,60 €, CAPI 0,00 €, total 733,60 €.",
    ).run(prompt)
    scored = score_stage(
        "imv", case, answer.answer,
        [r.model_dump(mode="json") for r in answer.tool_calls],
    )
    assert scored["selection_ok"] and scored["arguments_ok"]
    assert scored["tool_execution_ok"] and scored["oracle_ok"]
    assert scored["answer_money_ok"] and scored["success"]


def test_missing_facts_asks_instead_of_inventing_or_calling_tool():
    case = sample("imv", lambda c: c["category"] == "missing_facts")
    response = scripted(
        "imv", "Necesito conocer tu residencia legal y el patrimonio antes de calcularlo."
    ).run(case["query"])
    scored = score_stage("imv", case, response.answer, [])
    assert response.tool_calls == []
    assert scored["clarification_ok"] and scored["success"]


def test_invalid_imv_tool_call_is_observed_then_repaired():
    case = sample("imv", lambda c: c["category"] == "structured_calculation")
    first = block("calcular_imv", {"case": {"IMVCase": case["expected_arguments"]["case"]}})
    second = block("calcular_imv", case["expected_arguments"])
    response = scripted(
        "imv", first, second,
        "Simulación orientativa de IMV 733,60 euros, CAPI 0 euros, total 733,60 euros.",
    ).run(case["query"])
    assert not response.tool_calls[0].ok
    assert "invalid arguments" in response.tool_calls[0].error
    assert response.tool_calls[1].ok
    scored = score_stage(
        "imv", case, response.answer,
        [r.model_dump(mode="json") for r in response.tool_calls],
    )
    assert scored["had_schema_or_parse_error"] and scored["recovered_after_error"]
    assert scored["arguments_ok"] and scored["oracle_ok"]


def test_bdns_isolated_uses_real_http_interface_with_mock_network(monkeypatch):
    case = sample("bdns", lambda c: c["needs_tool"])
    expected = case["expected_arguments"]

    def fake_http(url, *, params, headers, timeout, follow_redirects):
        assert "infosubvenciones.es" in url
        assert params["descripcion"] == expected["tema"]
        return httpx.Response(
            200, json={
                "content": [{"numeroConvocatoria": 987654,
                             "descripcion": "SUBVENCIONES DE PRUEBA PARA ALQUILER"}],
                "totalElements": 1,
            }, request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "get", fake_http)
    result = scripted(
        "bdns", block("consultar_ayudas", expected),
        "La BDNS muestra SUBVENCIONES DE PRUEBA PARA ALQUILER; "
        "es una convocatoria, no una ayuda concedida.",
    ).run(case["query"])
    scores = score_stage(
        "bdns", case, result.answer,
        [r.model_dump(mode="json") for r in result.tool_calls],
    )
    assert scores["api_http_ok"] and scores["arguments_ok"]
    assert scores["source_grounding_heuristic"] and scores["success"]


def test_report_stage_observes_real_calculation_and_synthetic_bdns(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    case = sample("report", lambda c: c["api_fixture"] == "with_results")
    history = seed_report_observations(case)
    assert history[0].ok and history[0].name == "calcular_imv"
    assert history[1].ok and history[1].result["fixture"] is True

    messages_seen = []
    loop = ToolLoop(registry=registry_for_stage("report"))
    report_text = good_report(*history)
    def generate(messages):
        messages_seen.extend(messages)
        return block("guardar_informe", {"contenido": report_text})
    loop.generate = generate

    result = loop.run(case["query"], prior_observations=history)
    assert set(r.name for r in result.tool_calls) == {"guardar_informe"}
    assert result.tool_calls[0].ok
    assert [m["role"] for m in messages_seen] == [
        "system", "user", "assistant", "tool", "assistant", "tool"
    ]
    assert "calcular_imv" in messages_seen[3]["name"]
    assert "consultar_ayudas" in messages_seen[5]["name"]
    assert Path(result.tool_calls[0].result["archivo"]).read_text(encoding="utf-8") == report_text
    scores = score_stage(
        "report", case, result.answer,
        [r.model_dump(mode="json") for r in result.tool_calls],
        prior=history, report_root=tmp_path,
    )
    assert scores["file_verified"] and scores["file_sources_ok"]
    assert scores["file_amounts_ok"] and scores["success"]


def test_report_stage_error_observation_is_not_fake_grants(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    case = sample("report", lambda c: c["api_fixture"] == "error")
    history = seed_report_observations(case)
    assert not history[1].ok
    content = good_report(*history)
    response = scripted("report", block("guardar_informe", {"contenido": content})).run(
        case["query"], prior_observations=history,
    )
    scores = score_stage(
        "report", case, response.answer,
        [r.model_dump(mode="json") for r in response.tool_calls],
        prior=history, report_root=tmp_path,
    )
    assert scores["source_mode"] == "failed_api"
    assert scores["file_verified"] and scores["file_sources_ok"] and scores["success"]


def test_report_fails_metric_when_grant_sources_invented(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    case = sample("report", lambda c: c["api_fixture"] == "with_results")
    prior = seed_report_observations(case)
    content = good_report(*prior, cite=False)  # no actual observed call identifier
    result = scripted("report", block("guardar_informe", {"contenido": content})).run(
        case["query"], prior_observations=prior,
    )
    scores = score_stage(
        "report", case, result.answer,
        [r.model_dump(mode="json") for r in result.tool_calls],
        prior=prior, report_root=tmp_path,
    )
    assert scores["file_verified"]
    assert not scores["file_sources_ok"]
    assert not scores["success"]


def test_report_is_not_written_before_bdns_observation(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    case = sample("report")
    records = seed_report_observations(case)
    content = good_report(*records)
    result = scripted("report", block("guardar_informe", {"contenido": content}),
                      "No pude guardarlo.").run(
        case["query"], prior_observations=records[:1], max_turns=2,
    )
    assert not result.tool_calls[0].ok
    assert "BDNS" in result.tool_calls[0].error
    assert not list(tmp_path.glob("*.txt"))


def test_integration_parallel_then_report_preserves_order(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCA_IMV_REPORTS_DIR", str(tmp_path))
    case = sample("integration", lambda c: c["expected_tools"] == [
        "calcular_imv", "consultar_ayudas", "guardar_informe"
    ])
    from tool_use.tools import default_registry

    def fake_http(url, *, params, headers, timeout, follow_redirects):
        return httpx.Response(
            200, json={"content": [], "totalElements": 0},
            request=httpx.Request("GET", url),
        )
    monkeypatch.setattr(httpx, "get", fake_http)

    prior_calc = default_registry().call("calcular_imv", case["expected_arguments"])
    prior_api = default_registry().call("consultar_ayudas",
                                      case["expected_additional_arguments"]["consultar_ayudas"])
    assert prior_calc.ok and prior_api.ok
    report = good_report(prior_calc, prior_api)
    gen = (
        block("calcular_imv", case["expected_arguments"]) +
        block("consultar_ayudas", case["expected_additional_arguments"]["consultar_ayudas"])
    )
    out = scripted("integration", gen,
                   block("guardar_informe", {"contenido": report})).run(
        case["query"], max_turns=5,
    )
    assert [c.name for c in out.tool_calls] == [
        "calcular_imv", "consultar_ayudas", "guardar_informe"
    ]
    assert all(c.ok for c in out.tool_calls)
    metrics = score_workflow(
        case, out.answer, [c.model_dump(mode="json") for c in out.tool_calls],
        report_root=tmp_path,
    )
    assert metrics["multi_step_success"] and metrics["report_created_and_verified"]


def test_offline_stage_evaluator_returns_independent_denominators():
    case = sample("imv", lambda c: c["category"] == "missing_facts")
    loop = scripted("imv", "Necesito tu residencia legal y patrimonio.")
    output = evaluate_stage_cases([case], loop, "imv")
    assert output["summary"]["n_cases"] == 1
    assert output["summary"]["clarification_ok_n"] == 1
    assert output["summary"]["file_verified_n"] == 0


def test_bdns_empty_results_must_not_be_presented_as_grants(monkeypatch):
    case = sample("bdns", lambda c: c["needs_tool"])
    def empty_http(url, *, params, headers, timeout, follow_redirects):
        return httpx.Response(
            200, json={"content": [], "totalElements": 0},
            request=httpx.Request("GET", url),
        )
    monkeypatch.setattr(httpx, "get", empty_http)
    wrong = scripted(
        "bdns", block("consultar_ayudas", case["expected_arguments"]),
        "La BDNS ha encontrado tres convocatorias de alquiler.",
    ).run(case["query"])
    scores = score_stage(
        "bdns", case, wrong.answer,
        [c.model_dump(mode="json") for c in wrong.tool_calls],
    )
    assert scores["api_http_ok"]
    assert not scores["source_grounding_heuristic"]
    assert not scores["success"]
