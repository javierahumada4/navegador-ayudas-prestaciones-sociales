"""Independent phase-2 model evaluations mirroring phase-4 tool contexts.

IMV:      User natural language -> only calcular_imv (or a clarification).
BDNS:     User discovery request -> only consultar_ayudas (real HTTP).
REPORT:   User report request + trusted tool-role observations -> only guardar_informe.
INTEGRATION: Full registry, parallel first calls and dependent report action.

Report fixtures are EXPLICITLY SYNTHETIC: they are not actual BDNS data.
Use --report-api live to insert actual BDNS observations instead.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from api.schemas import ToolCallRecord
from tool_use.domain_tools import calcular_imv, consultar_ayudas, guardar_informe
from tool_use.evaluate import load_bench, normalise
from tool_use.executor import ToolLoop
from tool_use.imv_schema import IMVCase
from tool_use.metrics import (
    _file_status, expected_engine_result, explanation_checks, folded,
    money_fields_present, score_workflow,
)
from tool_use.registry import ToolRegistry
from tool_use.tools import default_registry
from tool_use.tracing import case_trace_path, create_artifact_run

STAGES = ("imv", "bdns", "report", "integration")
STAGE_TOOLS = {
    "imv": (calcular_imv,),
    "bdns": (consultar_ayudas,),
    "report": (guardar_informe,),
}


def registry_for_stage(stage: str) -> ToolRegistry:
    """Expose exactly the available actions, not three tools in every prompt."""
    if stage == "integration":
        return default_registry()
    if stage not in STAGE_TOOLS:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGES}")
    return ToolRegistry().register(*STAGE_TOOLS[stage])


def _fixture_bdns(case: dict[str, Any]) -> ToolCallRecord:
    """Synthetic and labelled observation to isolate report writing from HTTP."""
    kind = case.get("api_fixture", "with_results")
    args = {"tema": "alquiler", "max_resultados": 5}
    if kind == "error":
        return ToolCallRecord(
            name="consultar_ayudas", arguments=args, ok=False,
            error="BDNS unavailable (SIMULATED outage in offline evaluation)",
        )
    if kind == "empty":
        results = []
    elif kind == "with_results":
        results = [{
            "codigo_bdns": "SIMULADA-001",
            "titulo": "CONVOCATORIA FICTICIA DE AYUDA AL ALQUILER PARA EVALUACION",
            "fecha_registro": None, "administracion": "FICTICIA",
            "organismo": "ORGANISMO SIMULADO", "url": None,
        }]
    else:
        raise ValueError(f"Unknown api_fixture={kind!r}")
    return ToolCallRecord(
        name="consultar_ayudas", arguments=args, ok=True,
        result={
            "fuente": "FICTICIA: respuesta de prueba BDNS (NO convocatoria oficial)",
            "consulta": "alquiler",
            "resultados": results,
            "total_api": len(results),
            "advertencia": "Estos resultados son totalmente sintéticos; no existen tales ayudas.",
            "fixture": True,
        },
    )


def seed_report_observations(
    case: dict[str, Any], *, report_api: str = "fixture",
) -> list[ToolCallRecord]:
    """Generate tool observations using the REAL calculator, and fixture/live API.

    Only the already-completed tool records are inserted into the model chat,
    preserving assistant tool_call -> tool-role observation transitions.
    The model must author the report and invoke the file tool itself.
    """
    gold_case = case.get("expected_case")
    if not isinstance(gold_case, dict):
        raise ValueError("Report benchmark needs a normalized expected_case")
    calc = default_registry().call("calcular_imv", {"case": gold_case})
    if not calc.ok:
        raise ValueError(f"Broken report benchmark IMV fixture: {calc.error}")
    if report_api == "live":
        api = default_registry().call(
            "consultar_ayudas", {"tema": "alquiler", "max_resultados": 5}
        )
    elif report_api == "fixture":
        api = _fixture_bdns(case)
    else:
        raise ValueError("report_api must be fixture or live")
    return [calc, api]


def _gold_case_args_ok(expected: dict[str, Any], executed: list[dict]) -> bool:
    """Compare canonical Pydantic forms, including filled defaults."""
    expected_norm = IMVCase.model_validate(expected).model_dump(mode="json", by_alias=True)
    for tool_call in executed:
        if tool_call.get("name") != "calcular_imv" or not tool_call.get("ok"):
            continue
        try:
            original = tool_call["arguments"]["case"]
            parsed = IMVCase.model_validate(original).model_dump(mode="json", by_alias=True)
        except (ValueError, KeyError, TypeError):
            continue
        if parsed == expected_norm:
            return True
    return False


def _clarified(answer: str, keywords: list[str]) -> bool:
    text = folded(answer)
    if not any(word in text for word in ("necesito", "indica", "faltan", "falta", "dime", "facilita",
                                          "no puedo calcular", "no se puede", "datos", "proporciona")):
        return False
    return all(folded(item) in text for item in keywords)


def _status_oracle_matches(tool_calls: list[dict], gold: dict[str, Any]) -> bool:
    for call in tool_calls:
        if call.get("name") != "calcular_imv" or not call.get("ok"):
            continue
        result = call.get("result")
        if not isinstance(result, dict):
            continue
        relevant = (
            "imv_eligible", "capi_eligible", "status", "failed_requirements",
            "imv_monthly_eur", "capi_monthly_eur", "total_monthly_eur",
        )
        return all(result.get(key) == gold.get(key) for key in relevant)
    return False


def _source_grounded(text: str, observation: ToolCallRecord) -> bool:
    """Conservative heuristic for report sourcing; NOT a full factual verifier."""
    text_folded = folded(text)
    if "bdns" not in text_folded and "subvencion" not in text_folded and "convocatoria" not in text_folded:
        return False
    if not observation.ok:
        return bool(re.search(
            r"no (?:se )?(?:pudo|ha podido|fue posible)|"
            r"error|fallo|sin acceso|no disponible|indisponible",
            text_folded,
        ))
    results = (observation.result or {}).get("resultados", [])
    if not results:
        return bool(re.search(
            r"no se (?:han )?encontrado|sin resultados|"
            r"no hay convocatorias|ninguna convocatoria|0 resultados",
            text_folded,
        ))
    # At least one observed call identity must appear. A made-up code is rejected
    # when the model invents another identifier in the same synthetic namespace.
    seen_names = [
        folded(str(result.get("titulo") or "")) for result in results
    ]
    seen_ids = [
        folded(str(result.get("codigo_bdns") or "")) for result in results
    ]
    faithful = any(item and item in text_folded for item in seen_names + seen_ids)
    if observation.result.get("fixture") and "SIMULADA-" in text:
        observed = set(seen_ids)
        mentioned = set(re.findall(r"simulada-\d+", text_folded))
        faithful = faithful and mentioned <= observed
    return faithful


def score_stage(
    stage: str, case: dict[str, Any], answer: str, calls: list[dict],
    *, prior: list[ToolCallRecord] | None = None, report_root: Path | None = None,
) -> dict[str, Any]:
    """Auditable scores specific to the isolated tool's responsibility."""
    output: dict[str, Any] = {"stage": stage}
    expected_tool = case.get("expected_tool")
    first = calls[0] if calls else None
    output["selection_ok"] = (
        first is None if not case.get("needs_tool", True)
        else first is not None and first["name"] == expected_tool
    )
    output["tool_execution_ok"] = None
    output["arguments_ok"] = None
    output["clarification_ok"] = None
    output["oracle_ok"] = None
    output["answer_money_ok"] = None
    output["explanation_heuristic_ok"] = None
    output["file_verified"] = None
    output["file_amounts_ok"] = None
    output["file_sources_ok"] = None
    output["api_http_ok"] = None
    invalid_positions = [
        i for i, record in enumerate(calls)
        if not record.get("ok") and (
            "invalid arguments" in str(record.get("error", ""))
            or "Malformed tool call" in str(record.get("error", ""))
        )
    ]
    output["had_schema_or_parse_error"] = bool(invalid_positions)
    output["recovered_after_error"] = bool(
        invalid_positions and any(
            record.get("ok") for record in calls[min(invalid_positions) + 1:]
        )
    )
    output["false_save_claim"] = None

    if stage == "imv":
        successful = [
            c for c in calls if c["name"] == "calcular_imv" and c.get("ok")
        ]
        if case.get("needs_tool", True):
            output["tool_execution_ok"] = bool(successful)
            expected = case.get("expected_case") or case["expected_arguments"]["case"]
            output["arguments_ok"] = _gold_case_args_ok(expected, calls)
            oracle = expected_engine_result(case)
            output["oracle_ok"] = bool(oracle and _status_oracle_matches(calls, oracle))
            output["answer_money_ok"] = (
                all(money_fields_present(answer, oracle).values())
                if oracle and successful else False
            )
            output["explanation_heuristic_ok"] = (
                explanation_checks(answer, oracle)[0]
                if oracle and successful else False
            )
            output["success"] = bool(
                output["arguments_ok"] and output["oracle_ok"]
                and output["answer_money_ok"] and output["explanation_heuristic_ok"]
            )
        else:
            no_calculation = not any(c["name"] == "calcular_imv" for c in calls)
            output["clarification_ok"] = bool(
                no_calculation and _clarified(
                    answer, case.get("expected_clarifications", [])
                )
            )
            output["success"] = output["clarification_ok"]

    elif stage == "bdns":
        if not case.get("needs_tool", True):
            output["success"] = not calls
            return output
        successful = [c for c in calls if c["name"] == "consultar_ayudas" and c.get("ok")]
        attempted = [c for c in calls if c["name"] == "consultar_ayudas"]
        output["tool_execution_ok"] = bool(successful)
        output["api_http_ok"] = bool(successful)
        # Validation and HTTP availability are independent dimensions.
        output["arguments_ok"] = any(
            normalise(c.get("arguments", {})) == normalise(case["expected_arguments"])
            for c in attempted
        )
        # For errors, an honest failure is better than invented results.
        if successful:
            res = successful[-1].get("result") or {}
            results = res.get("resultados", []) if isinstance(res, dict) else []
            if not results:
                output["source_grounding_heuristic"] = bool(re.search(
                    r"(?:no (?:se )?(?:han )?encontrad|"
                    r"sin resultados|ninguna convocatoria|"
                    r"no hay convocatorias|0 resultados)",
                    folded(answer),
                ))
            else:
                output["source_grounding_heuristic"] = any(
                    folded(str(item.get("titulo", ""))) in folded(answer)
                    or folded(str(item.get("codigo_bdns", ""))) in folded(answer)
                    for item in results
                    if item.get("titulo") or item.get("codigo_bdns")
                )
        else:
            output["source_grounding_heuristic"] = bool(
                attempted and any(
                    word in folded(answer)
                    for word in ("error", "no he podido", "no se pudo", "sin acceso", "no disponible")
                )
            )
        output["success"] = bool(
            output["arguments_ok"] and output["api_http_ok"]
            and output["source_grounding_heuristic"]
        )

    elif stage == "report":
        prior = list(prior or [])
        if not any(r.ok and r.name == "calcular_imv" for r in prior):
            raise ValueError("Report stage requires a successful earlier calculation observation")
        api_records = [r for r in prior if r.name == "consultar_ayudas"]
        if not api_records:
            raise ValueError("Report stage requires the earlier BDNS observation")
        api = api_records[-1]
        output["source_mode"] = (
            "synthetic_fixture" if api.ok and api.result.get("fixture")
            else "live" if api.ok else "failed_api"
        )
        files = [c for c in calls if c["name"] == "guardar_informe" and c.get("ok")]
        output["tool_execution_ok"] = bool(files)
        output["false_save_claim"] = (
            bool(re.search(r"\b(?:guardado|creado)\s+correctamente\b", folded(answer)))
            and not files
        )
        ok, content = _file_status(files[-1] if files else None, report_root)
        output["file_verified"] = ok
        gold = expected_engine_result(case)
        if gold:
            output["file_amounts_ok"] = ok and all(
                money_fields_present(content, gold).values()
            )
            output["explanation_heuristic_ok"] = (
                explanation_checks(content, gold)[0] if ok else False
            )
        output["file_sources_ok"] = ok and _source_grounded(content, api)
        output["success"] = bool(
            output["selection_ok"] and output["file_verified"]
            and output["file_amounts_ok"] and output["file_sources_ok"]
            and output["explanation_heuristic_ok"]
            and not output["false_save_claim"]
        )

    elif stage == "integration":
        workflow = score_workflow(case, answer, calls, report_root=report_root)
        output["workflow"] = workflow
        if case.get("expected_behavior") == "partial_clarification":
            output["clarification_ok"] = (
                not any(c["name"] == "calcular_imv" and c.get("ok") for c in calls)
                and _clarified(answer, case.get("expected_clarifications", []))
            )
        else:
            output["clarification_ok"] = None
        output["tool_execution_ok"] = workflow["tool_sequence_success"]
        output["arguments_ok"] = workflow["all_tool_arguments_correct"]
        output["file_verified"] = workflow["report_created_and_verified"]
        output["answer_money_ok"] = workflow["amount_fidelity"]
        output["explanation_heuristic_ok"] = workflow["explanation_heuristic_accuracy"]
        output["success"] = bool(
            workflow["end_to_end_success"]
            and (output["clarification_ok"] is not False)
        )
    else:
        raise ValueError(f"Unknown stage {stage!r}")
    return output


def summarize_stage(rows: list[dict]) -> dict[str, Any]:
    fields = (
        "selection_ok", "tool_execution_ok", "arguments_ok", "clarification_ok",
        "oracle_ok", "answer_money_ok", "explanation_heuristic_ok",
        "api_http_ok", "source_grounding_heuristic", "file_verified",
        "file_amounts_ok", "file_sources_ok", "false_save_claim",
        "had_schema_or_parse_error", "recovered_after_error", "success",
    )
    summary: dict[str, Any] = {"n_cases": len(rows)}
    for key in fields:
        values = [row["scores"].get(key) for row in rows if row["scores"].get(key) is not None]
        summary[key] = sum(bool(value) for value in values) / len(values) if values else None
        summary[key + "_n"] = len(values)
    return summary


def evaluate_stage_cases(
    cases: list[dict[str, Any]], loop: ToolLoop, stage: str, *,
    max_turns: int = 5, report_api: str = "fixture", report_root: Path | None = None,
    artifact_run: Path | None = None,
) -> dict[str, Any]:
    """Execute one isolated stage, injectable with scripted generations for CPU tests."""
    rows = []
    for idx, case in enumerate(cases, 1):
        started = time.perf_counter()
        trace_path = (
            case_trace_path(artifact_run, stage, idx, case.get("id", f"case-{idx}"))
            if artifact_run is not None else None
        )
        prior = []
        try:
            prior = (
                seed_report_observations(case, report_api=report_api)
                if stage == "report" else []
            )
            response = loop.run(
                case["query"], max_turns=max_turns,
                prior_observations=prior or None,
                trace_path=trace_path,
            )
            answer = response.answer
            calls = [c.model_dump(mode="json") for c in response.tool_calls]
            score = score_stage(
                stage, case, answer, calls, prior=prior, report_root=report_root
            )
            row = {
                "id": case.get("id"), "query": case["query"],
                "answer": answer, "tool_calls": calls,
                "prior_observations": [
                    {"name": item.name, "ok": item.ok, "result": item.result,
                     "error": item.error}
                    for item in prior
                ] if stage == "report" else [],
                "turns": response.turns, "scores": score,
            }
        except Exception as exc:
            row = {
                "id": case.get("id"), "query": case["query"],
                "error": f"{type(exc).__name__}: {exc}",
                "tool_calls": [], "answer": "",
                "scores": {"stage": stage, "success": False},
            }
        row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        if trace_path is not None:
            row["trace_path"] = str(trace_path.resolve()) if trace_path.exists() else None
        rows.append(row)
        print(
            f"[{stage} {idx:02d}/{len(cases):02d}] "
            f"{case.get('id')} success={row['scores']['success']}",
            flush=True,
        )
    return {
        "stage": stage, "summary": summarize_stage(rows), "rows": rows,
        "api_observation_mode": report_api if stage == "report" else "live",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--model", default=os.getenv("ARCA_TOOLS_MODEL_ID", "Qwen/Qwen3-1.7B"))
    parser.add_argument("--brain", choices=["base", "rlm"], default="base")
    parser.add_argument("--adapter", default=os.getenv("ARCA_RLM_ADAPTER"))
    parser.add_argument("--report-api", choices=["fixture", "live"], default="fixture",
                        help="Only for REPORT: use labelled synthetic observations or live BDNS.")
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="reports/phase2_isolated.json")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")

    stages = STAGES if args.stage == "all" else (args.stage,)
    outcomes: dict[str, Any] = {
        "model": args.model, "brain": args.brain,
        "evaluation": "isolated_phase4_like",
        "report_api_default": args.report_api,
    }
    artifact_run = create_artifact_run(args.out)
    outcomes["artifacts_dir"] = str(artifact_run)
    outcomes["trace_format"] = "jsonl (one file per case, each event flushed immediately)"
    outcomes["reports_dir"] = str(artifact_run / "reports")
    shared_model = None
    old = os.environ.get("ARCA_IMV_REPORTS_DIR")
    os.environ["ARCA_IMV_REPORTS_DIR"] = str(artifact_run / "reports")
    try:
        for stage in stages:
            path = Path(__file__).with_name("bench") / f"{stage}.jsonl"
            cases = load_bench(path)
            if args.limit is not None:
                cases = cases[:args.limit]
            loop = ToolLoop(
                registry=registry_for_stage(stage), model_id=args.model,
                brain=args.brain, adapter_path=args.adapter,
            )
            if shared_model is not None:
                loop.model, loop.tokenizer = shared_model
            outcome = evaluate_stage_cases(
                cases, loop, stage, max_turns=args.max_turns,
                report_api=args.report_api,
                report_root=artifact_run / "reports",
                artifact_run=artifact_run,
            )
            if loop.model is not None:
                shared_model = loop.model, loop.tokenizer
            outcomes[stage] = outcome
    finally:
        if old is None:
            os.environ.pop("ARCA_IMV_REPORTS_DIR", None)
        else:
            os.environ["ARCA_IMV_REPORTS_DIR"] = old

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(outcomes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Trazas y archivos persistentes: {artifact_run}", flush=True)
    for stage in stages:
        print(stage, json.dumps(outcomes[stage]["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
