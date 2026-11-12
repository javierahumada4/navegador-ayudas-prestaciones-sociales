"""Discover dated IMV rulesets without guessing which law applies.

The file's effective interval denotes *declared* applicability. It does not
prove the law has not subsequently been amended. Source review is separate.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any


def ruleset_directory() -> Path:
    """Use the repository's rulesets directory (overridable in tests/deployments)."""
    return Path(os.environ.get("ARCA_RULESETS_DIR") or Path(__file__).with_name("rulesets"))


def inspect_rulesets(for_date: date) -> dict[str, Any]:
    """Find every file covering a date; report invalid files rather than trusting them."""
    directory = ruleset_directory()
    catalog: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    for path in sorted(directory.glob("imv_*.json")):
        try:
            rules = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rules, dict):
                raise ValueError("ruleset must be a JSON object")
            start = date.fromisoformat(rules["effective_from"])
            end = date.fromisoformat(rules["effective_to"])
            if end < start:
                raise ValueError("effective_to precedes effective_from")
            for section in ("amounts", "requirements", "patrimony", "capi", "monoparental"):
                if not isinstance(rules[section], dict):
                    raise ValueError(f"{section} must be a JSON object")
            if not isinstance(rules.get("id"), str) or not rules["id"]:
                raise ValueError("ruleset id is missing")
            catalog.append({
                "id": rules["id"],
                "file": str(path),
                "effective_from": start.isoformat(),
                "effective_to": end.isoformat(),
                "covers_date": start <= for_date <= end,
            })
        except (OSError, ValueError, KeyError, TypeError) as exc:
            invalid.append({"file": str(path), "error": str(exc)})
    matches = [row for row in catalog if row["covers_date"]]
    status = "vigente" if len(matches) == 1 else "ambiguo" if matches else "no_encontrado"
    # An invalid file may be a ruleset for this date. Never claim certainty.
    if invalid:
        status = "requiere_revision"
    return {
        "fecha_solicitud": for_date.isoformat(),
        "status": status,
        "ruleset": matches[0] if status == "vigente" else None,
        "disponibles": catalog,
        "archivos_invalidos": invalid,
        "advertencia": (
            "La cobertura por fechas del fichero no garantiza que las normas "
            "o las cuantias sigan actualizadas ante reformas posteriores."
        ),
    }


def ruleset_for_date(for_date: date) -> dict[str, Any]:
    """Return a unique dated ruleset, or fail closed without a 2026 fallback."""
    info = inspect_rulesets(for_date)
    if info["status"] != "vigente":
        raise ValueError(
            f"No hay un ruleset unico y valido para {for_date.isoformat()}: "
            f"{info['status']}. No se puede calcular sin normativa revisada."
        )
    return json.loads(Path(info["ruleset"]["file"]).read_text(encoding="utf-8"))
