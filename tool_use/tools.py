"""The tools your model can call. One is given as a worked example; the rest are yours.

``get_weather`` is the exact tool from the slides, using Open-Meteo (no API key). Study
how the docstring becomes the description and the type hints become the schema, then
write your own. Your domain needs at least three: one that queries the outside world
(a real HTTP API), one that computes or executes, and one that acts with an observable
effect. Ideas that have worked well: a currency or unit converter, a SQLite lookup, a
sandboxed Python evaluator, a calendar-event writer (``.ics`` file), a webhook notifier.

A tool that has side effects should be registered with ``requires_confirmation=True``;
in phase 4 you decide what the agent does with that flag.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from functools import lru_cache

import requests

from tool_use.registry import ToolRegistry, tool

OPEN_METEO_GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

# Open data API of the BOE (no key): consolidated, versioned text of Spanish laws.
BOE_CONSOLIDATED_API = "https://www.boe.es/datosabiertos/api/legislacion-consolidada/id"
LEY_IMV = "BOE-A-2021-21007"  # Ley 19/2021, de 20 de diciembre, del ingreso mínimo vital
BOE_ID_PATTERN = re.compile(r"BOE-A-\d{4}-\d{1,6}")
BOE_MAX_CHARS = 8000


def _normalize_title(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip().lower()


def _boe_title_query(articulo: str) -> str:
    """Normalise what the model asked ('13', 'art. 13', 'Anexo II') to a BOE index title."""
    query = _normalize_title(articulo)
    query = re.sub(r"^art(?:[íi]culo|\.)?\s*", "artículo ", query)
    if re.fullmatch(r"\d+(?: (?:bis|ter|quater))?", query):
        query = f"artículo {query}"
    return query


def resolve_boe_block(index: list[dict], articulo: str) -> dict:
    """Find the index entry for an article, annex or provision; also accepts a block id."""
    wanted = articulo.strip()
    for entry in index:
        if entry["id"] == wanted:
            return entry
    query = _boe_title_query(articulo)
    for entry in index:
        if _normalize_title(entry.get("titulo", "")) == query:
            return entry
    available = ", ".join(e["titulo"].replace("\xa0", " ") for e in index if e.get("titulo"))
    raise ValueError(f"no block {articulo!r} in this law. Available: {available}")


def parse_boe_block(xml_text: str) -> dict:
    """Return the version in force (the last one) of a BOE block as plain text."""
    root = ET.fromstring(xml_text)
    block = root.find("data/bloque")
    if block is None:
        raise ValueError("unexpected BOE response: no <bloque>")
    versions = block.findall("version")
    if not versions:
        raise ValueError("unexpected BOE response: block without versions")
    current = versions[-1]
    paragraphs = [re.sub(r"\s+", " ", "".join(p.itertext())).strip() for p in current]
    text = "\n".join(p for p in paragraphs if p)
    truncated = len(text) > BOE_MAX_CHARS
    return {
        "titulo": block.get("titulo", "").replace("\xa0", " "),
        "texto": text[:BOE_MAX_CHARS],
        "truncado": truncated,
        "vigente_desde": current.get("fecha_vigencia"),
        "modificado_por": current.get("id_norma"),
        "numero_versiones": len(versions),
    }


@lru_cache(maxsize=16)
def _boe_index(norma: str) -> tuple[dict, ...]:
    response = requests.get(
        f"{BOE_CONSOLIDATED_API}/{norma}/texto/indice",
        headers={"Accept": "application/json"},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()["data"]
    blocks = data["bloque"] if isinstance(data, dict) else data[0]["bloque"]
    return tuple(blocks)


@tool
def get_weather(location: str) -> float:
    """Return the current air temperature in degrees Celsius for a location.

    Args:
        location: City and country, for example 'Madrid, Spain'.
    """
    geo = requests.get(OPEN_METEO_GEOCODING, params={"name": location, "count": 1}, timeout=15)
    geo.raise_for_status()
    results = geo.json().get("results") or []
    if not results:
        raise ValueError(f"no coordinates found for {location!r}")
    lat, lon = results[0]["latitude"], results[0]["longitude"]
    forecast = requests.get(
        OPEN_METEO_FORECAST,
        params={
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "temperature_unit": "celsius",
        },
        timeout=15,
    )
    forecast.raise_for_status()
    return float(forecast.json()["current_weather"]["temperature"])


@tool
def calculator(expression: str) -> float:
    """Evaluate an arithmetic expression such as '(12.5 * 3) / 2 - 1'.

    Only numbers, parentheses and the operators + - * / ** % are allowed.

    Args:
        expression: The arithmetic expression to evaluate.
    """
    import ast
    import operator

    if len(expression) > 200:
        raise ValueError("expression is too long")

    allowed = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
    }

    def evaluate(node, depth=0):
        if depth > 25:
            raise ValueError("expression is too deeply nested")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in allowed:
            left = evaluate(node.left, depth + 1)
            right = evaluate(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and (abs(right) > 12 or abs(left) > 1e6):
                raise ValueError("exponentiation exceeds safe limits")
            result = allowed[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            result = allowed[type(node.op)](evaluate(node.operand, depth + 1))
        else:
            raise ValueError("unsupported expression")
        if not isinstance(result, (int, float)) or abs(result) > 1e15:
            raise ValueError("arithmetic result exceeds safe limits")
        return result

    return float(evaluate(ast.parse(expression, mode="eval").body))


@tool
def consultar_boe(articulo: str, norma: str = LEY_IMV) -> dict:
    """Devuelve el texto vigente de un artículo, anexo o disposición de una ley española,
    consultado en la API de datos abiertos del BOE (legislación consolidada).

    Úsala para citar o comprobar la base legal del Ingreso Mínimo Vital (IMV) y del
    Complemento de Ayuda para la Infancia (CAPI): requisitos, cuantía, límites de patrimonio...
    No calcula importes. Artículos útiles de la Ley 19/2021: 4-10 (beneficiarios y requisitos),
    11 (rentas computables), 13 (cuantía y renta garantizada), Anexo II (límite de patrimonio).
    El resultado incluye el texto, la fecha desde la que está vigente y la norma que lo modificó.

    Args:
        articulo: Bloque a consultar, por ejemplo '13', 'artículo 10', 'Anexo II' o
            'Disposición transitoria séptima'.
        norma: Identificador BOE de la ley; por defecto 'BOE-A-2021-21007' (Ley 19/2021, IMV).
    """
    if not BOE_ID_PATTERN.fullmatch(norma):
        raise ValueError("norma must look like 'BOE-A-2021-21007'")
    entry = resolve_boe_block(list(_boe_index(norma)), articulo)
    response = requests.get(
        f"{BOE_CONSOLIDATED_API}/{norma}/texto/bloque/{entry['id']}",
        headers={"Accept": "application/xml"},
        timeout=15,
    )
    response.raise_for_status()
    return {
        "norma": norma,
        **parse_boe_block(response.text),
        "url": f"https://www.boe.es/buscar/act.php?id={norma}",
    }


def default_registry() -> ToolRegistry:
    """The registry the executor and the agent use. Register your tools here."""
    registry = ToolRegistry().register(get_weather, calculator, consultar_boe)
    # Tu turno: registry.register(my_domain_api, my_domain_compute, my_domain_action)
    return registry
