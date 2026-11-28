"""Three domain tools: deterministic IMV/CAPI, live BDNS search, and text-file action.

The model reasons and composes reports; tools never ask a language model to calculate.
"""
import hashlib
import json
import os
import re
import secrets
from datetime import date
from pathlib import Path

import httpx
from pydantic import Field

from rlm.imv_engine import evaluate_imv_case
from tool_use.imv_schema import IMVCase
from tool_use.registry import tool

BDNS_ENDPOINT = "https://www.infosubvenciones.es/bdnstrans/api/convocatorias/busqueda"
RULESET_DIR = Path(__file__).resolve().parents[1] / "rlm" / "rulesets"


def rules_for_date(application_date):
    """Choose a checked, immutable ruleset for the given simulation date."""
    target = application_date.isoformat() if isinstance(application_date, date) else str(application_date)
    candidates = []
    for path in sorted(RULESET_DIR.glob("imv_*.json")):
        rules = json.loads(path.read_text(encoding="utf-8"))
        if rules["effective_from"] <= target <= rules["effective_to"]:
            candidates.append(rules)
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one validated IMV ruleset for {target}; found {len(candidates)}"
        )
    return candidates[0]


@tool
def calcular_imv(case: IMVCase) -> dict:
    """Calculate an IMV/CAPI *simulation* deterministically from normalized case facts.

    Use ONLY when sufficient dates, residency, household membership, financial figures
    and relevant family relationships are provided. Never invent missing facts:
    ask the user for them. This phase-1 engine covers common scenarios, not every
    administrative exception. The result is NOT an official eligibility decision.
    Do not calculate benefits yourself or modify returned numbers.

    Args:
        case: Complete IMV case with application_date, applicant_id, persons,
            domicile, economic facts, and relevant relationships/history.
            Dates must be YYYY-MM-DD and money is expressed in euros.
    """
    parsed = IMVCase.model_validate(case)
    normalized = parsed.model_dump(mode="json", by_alias=True)
    rules = rules_for_date(parsed.application_date)
    result = evaluate_imv_case(normalized, rules=rules)
    return {"ruleset_id": rules["id"], "application_date": normalized["application_date"], **result}


@tool
def consultar_ayudas(
    tema: str = Field(min_length=3, max_length=100),
    max_resultados: int = Field(default=5, ge=1, le=20),
) -> dict:
    """Search actual public grant calls through Spain's official BDNS HTTP API.

    Use to discover additional public grants (housing, families, disability, etc.).
    This is NOT an IMV/CAPI calculator, and the BDNS does not cover all social
    benefits. Results are not proof of eligibility or of an open application period.

    Args:
        tema: Keywords appearing in the grant call title (e.g. 'alquiler').
        max_resultados: Between 1 and 20 matching calls to return.
    """
    term = tema.strip()
    if len(term) < 3 or len(term) > 100:
        raise ValueError("tema must have between 3 and 100 characters")
    if not 1 <= max_resultados <= 20:
        raise ValueError("max_resultados must be between 1 and 20")
    params = {
        "descripcion": term,
        "descripcionTipoBusqueda": 1,
        "page": 0,
        "pageSize": max_resultados,
        "order": "fechaRecepcion",
        "direccion": "desc",
    }
    try:
        response = httpx.get(
            BDNS_ENDPOINT, params=params,
            headers={"Accept": "application/json", "User-Agent": "ARCA-IMV-Research/1.0"},
            timeout=15.0, follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(f"BDNS unavailable or invalid response: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("content"), list):
        raise ValueError("BDNS response has unexpected structure")
    found = []
    for item in payload["content"][:max_resultados]:
        if not isinstance(item, dict):
            continue
        number = item.get("numeroConvocatoria")
        number = str(number) if number is not None else None
        found.append({
            "codigo_bdns": number,
            "titulo": str(item.get("descripcion") or "")[:500],
            "fecha_registro": item.get("fechaRecepcion"),
            "administracion": item.get("nivel1"),
            "organismo": item.get("nivel2"),
            "url": (
                f"https://www.infosubvenciones.es/bdnstrans/GE/es/convocatoria/{number}"
                if number and re.fullmatch(r"[0-9]+", number) else None
            ),
        })
    return {
        "fuente": "BDNS - Sistema Nacional de Publicidad de Subvenciones",
        "consulta": term,
        "resultados": found,
        "total_api": payload.get("totalElements"),
        "advertencia": (
            "Convocatoria encontrada no implica plazo abierto ni elegibilidad. "
            "Consultar bases, requisitos, fechas y organismos."
        ),
    }


@tool(requires_confirmation=True)
def guardar_informe(contenido: str = Field(min_length=80, max_length=20000)) -> dict:
    """Write the report *already composed by the language model* to a UTF-8 .txt file.

    Call only after the user explicitly requests saving a report and the model has
    described the real calculation/search observations faithfully. NEVER pass a file
    path or identifiers. This tool does not draft, rewrite or calculate anything.

    Args:
        contenido: Full natural-language report, including eligibility, accurate
            engine amounts, reasons, caveats and any BDNS sources as applicable.
    """
    if not 80 <= len(contenido) <= 20000:
        raise ValueError("report must contain 80-20000 characters")
    directory = Path(os.environ.get("ARCA_IMV_REPORTS_DIR", "reports/imv")).expanduser()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    for _ in range(10):
        path = directory / f"informe-imv-{secrets.token_hex(12)}.txt"
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(contenido)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return {
            "archivo": str(path.resolve()),
            "bytes": len(contenido.encode("utf-8")),
            "sha256": hashlib.sha256(contenido.encode("utf-8")).hexdigest(),
        }
    raise RuntimeError("Unable to allocate a unique report path")
