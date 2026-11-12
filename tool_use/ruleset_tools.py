"""Read-only date/BOE/ruleset discovery plus a confirmation-gated ruleset writer.

The BOE API provides legal documents, NOT a complete numeric IMV ruleset.
A proposal must be checked against primary BOE and Social Security sources
before an approved operator can publish it for the deterministic engine.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import httpx

from rlm.ruleset_catalog import inspect_rulesets, ruleset_directory
from tool_use.registry import tool

BOE_BASE_URL = "https://www.boe.es/datosabiertos/api/legislacion-consolidada/id"
BOE_ID_PATTERN = re.compile(r"BOE-A-[0-9]{4}-[0-9]+\Z")
MAX_BOE_CHARS = 12000
_DECIMAL_FIELDS = {
    "amounts": (
        "single_adult_monthly_eur", "additional_member_pct",
        "max_household_multiplier", "monoparental_pct", "disability_pct",
        "disability_min_percent", "minimum_imv_monthly_eur",
    ),
    "requirements": (
        "minimum_continuous_residence_months", "minimum_household_formation_months",
        "individual_min_age", "independence_under_age",
        "independence_under_age_months", "independence_under_age_ss_months",
        "independence_age_30_plus_months",
    ),
    "patrimony": (
        "single_adult_net_worth_multiplier_annual",
        "single_adult_non_corporate_assets_multiplier_annual",
    ),
    "capi": ("income_multiplier", "net_worth_multiplier"),
    "monoparental": ("other_parent_institution_min_months", "dependency_min_grade"),
}
_ANNEX_KEYS = {
    "1,0", "1,1", "1,2", "1,3", "1,4+", "2,0", "2,1", "2,2",
    "2,3+", "3,0", "3,1", "3,2+", "4,0", "4,1+", "other",
}


@tool
def fecha_actual() -> dict:
    """Return today's calendar date in Europe/Madrid for a request made today.

    Never substitute today's date if the user provided an application date.
    Returns:
        fecha_actual and timezone for interpreting current-date requests.
    """
    return {"fecha_actual": datetime.now(ZoneInfo("Europe/Madrid")).date().isoformat(),
            "timezone": "Europe/Madrid"}


@tool
def comprobar_ruleset(fecha_solicitud: date) -> dict:
    """Check existing local IMV/CAPI rulesets by their effective date intervals.

    Call before BOE lookup. Use the actual application date, not today's date
    except for a new application explicitly made today. A date-covered
    ruleset does not independently prove the law has not changed.

    Args:
        fecha_solicitud: Application date in ISO format YYYY-MM-DD.
    """
    return inspect_rulesets(fecha_solicitud)


@tool
def consultar_boe(
    boe_id: str = "BOE-A-2021-21007",
    seccion: Literal["metadatos", "analisis", "texto"] = "metadatos",
) -> dict:
    """Query the REAL BOE open-data REST API for legislation on IMV.

    Use only if a date-appropriate ruleset is missing, ambiguous, needs review,
    or the user explicitly requests up-to-date legal sources. This API does not
    supply a ready-made JSON of IMV/CAPI amounts. Inspect the legal text and
    official annual amounts separately before proposing numeric rules.

    Args:
        boe_id: Official BOE identifier such as BOE-A-2021-21007.
        seccion: Metadata, legal analysis or consolidated text; text is truncated.
    """
    if not BOE_ID_PATTERN.fullmatch(boe_id):
        raise ValueError("BOE id must have format BOE-A-YYYY-NNNNN")
    url = f"{BOE_BASE_URL}/{boe_id}/{seccion}"
    headers = {"Accept": "application/json" if seccion != "texto" else "application/xml"}
    response = httpx.get(url, headers=headers, timeout=15.0, follow_redirects=False)
    response.raise_for_status()
    if len(response.content) > 2_000_000:
        raise ValueError("BOE response is too large for tool use")
    if seccion == "texto":
        body = response.text
        return {
            "boe_id": boe_id,
            "seccion": seccion,
            "api_url": url,
            "reference_url": f"https://www.boe.es/buscar/act.php?id={boe_id}",
            "texto": body[:MAX_BOE_CHARS],
            "truncado": len(body) > MAX_BOE_CHARS,
            "nota": "Texto informativo. Verificar las versiones aplicables a la fecha solicitada.",
        }
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Unexpected BOE API response format")
    if payload.get("status") in {"ERROR", "KO"}:
        raise ValueError(f"BOE API reports an error: {payload.get('message', 'unknown')}")
    return {
        "boe_id": boe_id,
        "seccion": seccion,
        "api_url": url,
        "reference_url": f"https://www.boe.es/buscar/act.php?id={boe_id}",
        "respuesta": payload,
        "nota": "La vigencia legal y las cuantias anuales requieren comprobacion documental.",
    }


def _official_domain(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Sources must use HTTPS official domains")
    host = parsed.hostname.lower()
    allowed = (
        host == "boe.es" or host.endswith(".boe.es")
        or host == "seg-social.es" or host.endswith(".seg-social.es")
    )
    if not allowed:
        raise ValueError(f"Non-official source domain: {host}")
    return "boe" if host == "boe.es" or host.endswith(".boe.es") else "seg-social"


def _positive_number(value: Any, label: str, *, strictly: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{label} must be numeric")
    try:
        n = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not n.is_finite() or n < 0 or (strictly and n <= 0):
        raise ValueError(f"{label} must be finite and non-negative")


def _validate_proposal(ruleset: dict[str, Any]) -> tuple[date, date]:
    if not isinstance(ruleset, dict):
        raise ValueError("ruleset must be a complete JSON object")
    if not isinstance(ruleset.get("id"), str) or not re.fullmatch(
        r"imv_es_[0-9]{4}(?:_[0-9a-z_]+)?", ruleset["id"]
    ):
        raise ValueError("ruleset id must start imv_es_YYYY")
    start, end = date.fromisoformat(ruleset["effective_from"]), date.fromisoformat(
        ruleset["effective_to"]
    )
    if start > end:
        raise ValueError("effective_from must precede effective_to")
    if start.year != end.year:
        raise ValueError("Publish separate rulesets for each calendar year")
    if ruleset["id"].split("_")[2] != str(start.year):
        raise ValueError("ruleset ID year must match effective_from")
    for section, fields in _DECIMAL_FIELDS.items():
        data = ruleset.get(section)
        if not isinstance(data, dict):
            raise ValueError(f"Missing {section} section")
        for field in fields:
            _positive_number(data[field], f"{section}.{field}")
    _positive_number(ruleset["amounts"]["single_adult_monthly_eur"],
                     "single_adult_monthly_eur", strictly=True)
    _positive_number(ruleset["amounts"]["max_household_multiplier"],
                     "max_household_multiplier", strictly=True)
    annex = ruleset["patrimony"].get("annex_ii")
    if not isinstance(annex, dict) or not _ANNEX_KEYS.issubset(annex):
        raise ValueError("Incomplete patrimony.annex_ii")
    for key in _ANNEX_KEYS:
        _positive_number(annex[key], f"annex_ii.{key}", strictly=True)
    bands = ruleset["capi"].get("age_bands")
    if not isinstance(bands, list) or not bands:
        raise ValueError("Missing capi.age_bands")
    last = 0
    for band in bands:
        if not isinstance(band, dict) or type(band.get("max_age_exclusive")) is not int:
            raise ValueError("Invalid CAPI age band")
        limit = band["max_age_exclusive"]
        if limit <= last or limit > 18:
            raise ValueError("CAPI age bands must be increasing and end at 18")
        _positive_number(band["monthly_eur"], "capi.monthly_eur")
        last = limit
    if last != 18:
        raise ValueError("CAPI age bands must cover all minors up to 18")
    degrees = ruleset["monoparental"].get("qualifying_permanent_disability_degrees")
    if not isinstance(degrees, list) or not all(isinstance(d, str) for d in degrees):
        raise ValueError("Invalid qualifying disability degrees")
    sources = ruleset.get("sources")
    if not isinstance(sources, list) or len(sources) < 2:
        raise ValueError("Need at least BOE and Seguridad Social sources")
    domains = set()
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("name"), str):
            raise ValueError("Each source needs name, url and checked date")
        domains.add(_official_domain(source["url"]))
        date.fromisoformat(source["checked"])
    if not {"boe", "seg-social"}.issubset(domains):
        raise ValueError("Need both BOE law and official Seguridad Social amounts")
    return start, end


@tool(requires_confirmation=True)
def crear_ruleset(
    ruleset: dict[str, Any],
    confirmacion_humana: bool = False,
) -> dict:
    """Publish a FULL, independently reviewed dated ruleset for the IMV engine.

    Side effect: writes a local JSON file. Never run without HUMAN confirmation;
    the phase-2 tool loop blocks calls to this tool regardless of arguments.
    BOE text alone cannot establish the numeric annual amounts. Do not copy
    another year's amounts or invent legal thresholds. Every value must come
    from reviewed official sources and the complete JSON must match the engine
    schema. The file is never overwritten and periods may not overlap.

    Args:
        ruleset: Complete proposed engine JSON including effective dates, all
            numerical parameters, and BOE/Seguridad Social URLs + check dates.
        confirmacion_humana: True only when an operator has reviewed every amount
            and explicitly authorised this specific write.
    """
    if confirmacion_humana is not True:
        raise PermissionError("Human confirmation required to publish a ruleset")
    start, end = _validate_proposal(ruleset)
    directory = ruleset_directory()
    current = inspect_rulesets(start)
    if current["archivos_invalidos"]:
        raise ValueError("Cannot publish while the existing catalog has invalid files")
    for entry in current["disponibles"]:
        a = date.fromisoformat(entry["effective_from"])
        b = date.fromisoformat(entry["effective_to"])
        if start <= b and a <= end:
            raise ValueError(f"Ruleset period overlaps existing {entry['file']}")
    name = f"imv_{start.year}.json" if (
        start == date(start.year, 1, 1) and end == date(end.year, 12, 31)
    ) else f"imv_{start.isoformat()}_{end.isoformat()}.json"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / name
    if destination.exists():
        raise FileExistsError(str(destination))
    # Exclusive creation: never replace a pre-existing ruleset.
    payload = (json.dumps(ruleset, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with destination.open("xb") as handle:
        handle.write(payload)
    return {
        "created": True,
        "file": str(destination),
        "effective_from": start.isoformat(),
        "effective_to": end.isoformat(),
        "reviewed": True,
    }
