"""End-to-end, independently auditable phase-2 scoring.

The teacher's first-tool/arguments score remains unchanged. These additional
metrics cover whole workflows and actual on-disk files. Explanations are
measured with *declared heuristics*, not assumed to be legally validated.
"""
from __future__ import annotations

import hashlib
from collections import Counter
import re
import unicodedata
from pathlib import Path
from typing import Any

from rlm.imv_engine import evaluate_imv_case
from tool_use.domain_tools import rules_for_date
from tool_use.executor import _money_is_mentioned

FAILURE_WORDS = {
    "RESIDENCE_NOT_MET": ("residenc",),
    "APPLICANT_AGE_NOT_MET": ("edad", "menor"),
    "INDEPENDENCE_REQUIREMENT_NOT_MET": ("independenc",),
    "HOUSEHOLD_FORMATION_PERIOD_NOT_MET": ("convivenc", "constituci", "antiguedad"),
    "PERMANENT_RESIDENTIAL_SERVICE_EXCLUSION": ("residencial", "centro permanente"),
    "ACTIVE_COMPANY_DIRECTOR": ("administrador", "sociedad", "empresa"),
    "NET_WORTH_LIMIT_EXCEEDED": ("patrimonio",),
    "NON_CORPORATE_ASSET_LIMIT_EXCEEDED": ("activo", "bienes"),
    "IMV_INCOME_LIMIT_EXCEEDED": ("ingreso", "renta"),
    "PENSION_CAP_EXCLUDES_IMV": ("pension",),
    "CAPI_INCOME_LIMIT_EXCEEDED": ("ingreso", "renta"),
    "CAPI_NET_WORTH_LIMIT_EXCEEDED": ("patrimonio",),
    "CAPI_NON_CORPORATE_ASSET_LIMIT_EXCEEDED": ("activo", "bienes"),
}


def folded(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def expected_engine_result(case: dict[str, Any]) -> dict[str, Any] | None:
    """Compute gold independently from model output and actual tool execution."""
    raw = case.get("expected_case")
    if raw is None:
        raw = case.get("expected_arguments", {}).get("case")
    if not raw:
        return None
    try:
        return evaluate_imv_case(raw, rules_for_date(raw["application_date"]))
    except (ValueError, KeyError, TypeError):
        return None


def money_fields_present(text: str, gold: dict[str, Any]) -> dict[str, bool]:
    """Strict values to the cent (Spanish and English formatting accepted).

    This is a lexical number check, not a semantic verification of labels.
    In particular, coincidental numbers in prose can produce false positives.
    """
    return {
        key: _money_is_mentioned(text, gold[key])
        for key in ("imv_monthly_eur", "capi_monthly_eur", "total_monthly_eur")
    }


def _file_status(record: dict[str, Any] | None, root: Path | None) -> tuple[bool, str]:
    if not record:
        return False, ""
    result = record.get("result")
    if not isinstance(result, dict) or not result.get("archivo"):
        return False, ""
    try:
        path = Path(result["archivo"]).resolve(strict=True)
        if root is not None and not path.is_relative_to(root.resolve()):
            return False, ""
        if path.suffix.lower() != ".txt" or not path.is_file():
            return False, ""
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != result.get("sha256"):
            return False, ""
        if len(content) != result.get("bytes"):
            return False, ""
        text = content.decode("utf-8")
        if text != record.get("arguments", {}).get("contenido"):
            return False, ""
        return True, text
    except (OSError, UnicodeDecodeError, ValueError, TypeError):
        return False, ""


def explanation_checks(text: str, gold: dict[str, Any]) -> tuple[bool, float]:
    """Lightweight contradiction/reason checks; not legal correctness certification."""
    normalized = folded(text)
    codes = gold["failed_requirements"]
    covered = sum(
        any(word in normalized for word in FAILURE_WORDS.get(code, (folded(code),)))
        for code in codes
    )
    coverage = covered / len(codes) if codes else 1.0
    terms_ok = "imv" in normalized and "capi" in normalized
    caveat_ok = any(
        word in normalized for word in (
            "simulacion", "orientativ", "estimacion", "no es una resolucion",
            "no constituye una resolucion", "no representa una decision",
        )
    )
    false_approval = (
        gold["status"] == "ineligible"
        and any(phrase in normalized for phrase in (
            "es elegible para ambos", "cumple todos los requisitos",
            "tiene derecho a ambas prestaciones",
        ))
    )
    return bool(terms_ok and caveat_ok and coverage == 1 and not false_approval), coverage


def score_workflow(
    case: dict[str, Any], answer: str, tool_calls: list[dict[str, Any]],
    *, report_root: Path | None = None,
) -> dict[str, Any]:
    """Score a single complete trace, including confirmations and file integrity."""
    required = case.get("expected_tools")
    if required is None:
        required = [case["expected_tool"]] if case.get("needs_tool", True) else []
    successful = [
        c["name"] for c in tool_calls if c.get("ok")
    ]
    required_ok = not (Counter(required) - Counter(successful))
    if not required:
        required_ok = not tool_calls
    if "guardar_informe" in required:
        # A report must follow an observed calculation, not precede it.
        try:
            required_ok = required_ok and (
                successful.index("calcular_imv") < successful.index("guardar_informe")
            )
        except ValueError:
            required_ok = False
    if "guardar_informe" not in required and "guardar_informe" in successful:
        required_ok = False

    report_record = next(
        (c for c in tool_calls if c.get("ok") and c["name"] == "guardar_informe"), None
    )
    file_exists, report_text = _file_status(report_record, report_root)
    expects_report = "guardar_informe" in required

    gold = expected_engine_result(case)
    amount_accuracy = None
    explanation_accuracy = None
    reason_coverage = None
    report_amount_accuracy = None
    if gold is not None:
        values = money_fields_present(answer, gold)
        # In report cases, the final answer need only confirm the file; the
        # generated file itself is the artifact that must carry accurate amounts.
        if expects_report:
            amount_accuracy = all(money_fields_present(report_text, gold).values()) if file_exists else False
            report_amount_accuracy = amount_accuracy
        else:
            amount_accuracy = all(values.values())
        check_text = report_text if expects_report and file_exists else answer
        explanation_accuracy, reason_coverage = explanation_checks(check_text, gold)

    clarification_accuracy = None
    if case.get("expected_behavior") == "ask_clarification":
        normal = folded(answer)
        has_request = any(
            word in normal for word in ("necesito", "falta", "indica", "facilita", "podrias", "dime", "datos")
        )
        mentions_all = all(
            folded(keyword) in normal for keyword in case.get("expected_clarifications", [])
        )
        clarification_accuracy = bool(not tool_calls and has_request and mentions_all)

    unverified_save_claim = bool(
        not report_record and re.search(
            r"(?i)(?:he|hemos|se ha|esta)\s+(?:guardado|creado|generado)"
            r".{0,70}(?:archivo|fichero|informe)"
            r"|(?:informe|archivo|fichero).{0,70}(?:guardado|creado correctamente)",
            folded(answer),
        )
    )

    task_success = bool(required_ok)
    if expects_report:
        task_success = task_success and file_exists
    if gold is not None:
        task_success = task_success and bool(amount_accuracy) and bool(explanation_accuracy)
    if clarification_accuracy is not None:
        task_success = task_success and clarification_accuracy
    task_success = task_success and not unverified_save_claim

    return {
        "required_tools": required,
        "tool_sequence_success": bool(required_ok),
        "multi_step_success": bool(required_ok) if len(required) > 1 else None,
        "amount_fidelity": amount_accuracy,
        "report_created_and_verified": file_exists if expects_report else None,
        "report_amount_fidelity": report_amount_accuracy,
        "explanation_heuristic_accuracy": explanation_accuracy,
        "failed_reason_coverage": reason_coverage,
        "clarification_accuracy": clarification_accuracy,
        "unverified_save_claim": unverified_save_claim,
        "end_to_end_success": task_success,
        "expected_engine": gold,
    }


def summarise_workflows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = (
        "tool_sequence_success", "multi_step_success", "amount_fidelity",
        "report_created_and_verified", "report_amount_fidelity",
        "explanation_heuristic_accuracy", "failed_reason_coverage",
        "clarification_accuracy", "end_to_end_success",
    )
    summary = {}
    for name in names:
        values = [r["workflow"][name] for r in rows if r["workflow"][name] is not None]
        summary[name] = sum(float(value) for value in values) / len(values) if values else None
        summary[name + "_n"] = len(values)
    summary["unverified_save_claims"] = sum(
        r["workflow"]["unverified_save_claim"] for r in rows
    )
    return summary
