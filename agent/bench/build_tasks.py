"""Build ``agent/bench/tasks.jsonl``: 20 tasks that chain at least two tools, plus 3 impossible.

Expected amounts are computed here with the deterministic engine (``calcular_imv``), never
typed by hand, so the benchmark is auditable: ``uv run python -m agent.bench.build_tasks``
regenerates the same file. Legal facts come from the phase-3 gold set (``rag/gold/gold.jsonl``,
evidence quoted literally from the corpus): minimum age 23 (g02), 6 months of household (g07),
10 euros of margin (g08), 6 months to resolve (g24), 30 days to report changes (g26), net-worth
limit 26 409,60 EUR (g20), asset limit 8 803,20 EUR (g21), CAPI 115 EUR under 3 in 2026 (g17).

Cases are given as normalised JSON in most tasks (what the engine needs; phase 2 showed the
1.7B model fails 0/6 at extracting prose cases) and in prose in three tasks, to measure that
failure in the agent too. Check types (``agent/evaluate.py``): ``number`` (Spanish or English
formatting), ``contains``, ``regex``, and ``all`` / ``any`` of several checks.

Checks corrected after the baseline scorecard (2026-11-24), where right answers were scored as
wrong: a denial written "es ineligible", "seis meses" in words, and IMV and CAPI given separately
instead of the total. ``pipeline_eval/compare.py`` re-scores stored answers of both variants
with the current checks, so the correction applies equally to baseline and fixes.
"""

from __future__ import annotations

import json
from pathlib import Path

from tool_use.domain_tools import calcular_imv

OUT = Path(__file__).with_name("tasks.jsonl")
ABSTAIN = r"(?i)no puedo|no es posible|no se puede|falta|faltan|necesito|no dispongo|no consta|no aparece|no encuentro|no tengo"
DENIED = (r"(?i)no (?:tiene|tendría|le corresponde|cumple|reúne)|sin derecho|deneg|"
          r"no es (?:elegible|beneficiari)|in(?:e|é)l[ie]gible|ineligible|no elegible")


def person(pid: str, born: str, since: str) -> dict:
    return {
        "id": pid, "date_of_birth": born, "legal_residence_in_spain": True,
        "effective_residence_in_spain": True, "continuous_legal_effective_residence_since": since,
    }


def case(persons, relationships=(), income=0, worth=0, assets=0, date="2026-06-01") -> dict:
    ids = [p["id"] for p in persons]
    return {
        "application_date": date,
        "applicant_id": ids[0],
        "persons": list(persons),
        "domicile": {"co_resident_ids": ids, "same_domicile_since": "2020-01-01"},
        "relationships": list(relationships),
        "applicant_history": {"independent_from_parents_since": "2010-01-01"},
        "economic": {
            "countable_income_annual_eur": income, "net_worth_eur": worth,
            "non_corporate_assets_eur": assets,
        },
    }


def child_of(parent: str, child: str, custody: str = "exclusive") -> dict:
    return {"person_a": parent, "person_b": child, "type": "parent_child", "custody": custody}


CASES = {
    "A": case([person("adult", "1980-01-01", "2020-01-01")]),
    "B": case(
        [person("adult", "1985-04-02", "2018-01-01"), person("child", "2020-01-01", "2020-01-01")],
        [child_of("adult", "child")], income=3000,
    ),
    "C": case(
        [person("p1", "1982-02-01", "2015-01-01"), person("p2", "1984-07-15", "2015-01-01"),
         person("k1", "2014-03-01", "2015-01-01"), person("k2", "2017-09-10", "2017-09-10")],
        [{"person_a": "p1", "person_b": "p2", "type": "spouse"},
         child_of("p1", "k1", "shared"), child_of("p2", "k1", "shared"),
         child_of("p1", "k2", "shared"), child_of("p2", "k2", "shared")],
        income=6000,
    ),
    "D": case([person("adult", "1979-05-05", "2012-01-01")], income=12000),
    "E": case(
        [person("mother", "1992-08-08", "2016-01-01"), person("baby", "2025-02-01", "2025-02-01")],
        [child_of("mother", "baby")], income=2000,
    ),
    "F": case([person("adult", "1970-10-10", "2010-01-01")], worth=30000),
}


def amounts(key: str) -> dict:
    result = calcular_imv(case=CASES[key])
    return {k: result[k] for k in ("status", "imv_monthly_eur", "capi_monthly_eur", "total_monthly_eur")}


def cjson(key: str) -> str:
    return json.dumps(CASES[key], ensure_ascii=False, separators=(",", ":"))


def number(value: float, tolerance: float = 0.01) -> dict:
    return {"type": "number", "value": round(value, 2), "tolerance": tolerance}


def total_or_denied(key: str) -> dict:
    """The total, or IMV and CAPI given separately (both are a correct answer), or a denial."""
    result = amounts(key)
    total = result["total_monthly_eur"]
    if total > 0:
        if result["capi_monthly_eur"] > 0:
            parts = {"type": "all", "checks": [number(result["imv_monthly_eur"]),
                                               number(result["capi_monthly_eur"])]}
            return {"type": "any", "checks": [number(total), parts]}
        return number(total)
    return {"type": "any", "checks": [number(0), {"type": "regex", "value": DENIED}]}


def tasks() -> list[dict]:
    A, B, C, D, E = (amounts(k) for k in "ABCDE")
    kb, calc, boe, bdns, save = (
        "search_knowledge_base", "calcular_imv", "consultar_boe", "consultar_ayudas", "guardar_informe"
    )
    rows = [
        ("T01", "kb+calc", f"Busca en la base de conocimiento la edad mínima para ser titular del IMV y calcula con el motor este caso: {cjson('A')}. Dime la edad mínima y el total mensual.",
         [kb, calc], {"type": "all", "checks": [number(23), number(A["total_monthly_eur"])]}),
        ("T02", "boe+calc", f"Consulta en el BOE el artículo 13 de la Ley 19/2021 (cuantía) y calcula con el motor este caso: {cjson('B')}. Dame el total mensual.",
         [boe, calc], total_or_denied("B")),
        ("T03", "calc+bdns", f"Calcula el IMV/CAPI de este caso: {cjson('C')} y busca 3 convocatorias de ayudas al alquiler en la BDNS. Dame el total mensual y el título de una convocatoria.",
         [calc, bdns], total_or_denied("C")),
        ("T04", "calc+save", f"Calcula el IMV de este caso: {cjson('A')} y guarda un informe TXT con el resultado.",
         [calc, save], {"type": "all", "checks": [number(A["total_monthly_eur"]), {"type": "regex", "value": r"(?i)guard|informe"}]}),
        ("T05", "kb+calc", f"Busca en la base de conocimiento en cuántos días hay que comunicar un cambio a la Seguridad Social y calcula este caso: {cjson('D')}. Dime los días y el resultado del cálculo.",
         [kb, calc], {"type": "all", "checks": [number(30), total_or_denied("D")]}),
        ("T06", "kb+calc", f"Busca en la base de conocimiento cuánto es el complemento de ayuda para la infancia de un menor de tres años en 2026 y calcula este caso: {cjson('E')}. Dame el CAPI y el total mensual.",
         [kb, calc], total_or_denied("E")),
        ("T07", "kb+calc", f"Busca en la base de conocimiento el límite de patrimonio neto de una persona que vive sola en 2026 y calcula este caso: {cjson('F')}. ¿Tiene derecho al IMV?",
         [kb, calc], {"type": "all", "checks": [number(26409.60), total_or_denied("F")]}),
        ("T08", "boe+kb", "Consulta en el BOE el artículo 10 de la Ley 19/2021 y busca en la base de conocimiento cuántos meses debe llevar constituida la unidad de convivencia antes de la solicitud.",
         [boe, kb], {"type": "any", "checks": [number(6), {"type": "regex", "value": r"(?i)\bseis meses"}]}),
        ("T09", "kb+bdns", "¿Es compatible el IMV con las rentas del trabajo? Búscalo en la base de conocimiento y después busca 2 convocatorias de empleo en la BDNS.",
         [kb, bdns], {"type": "regex", "value": r"(?i)compatib"}),
        ("T10", "calc+kb", f"Calcula con el motor el caso 1: {cjson('A')} y el caso 2: {cjson('B')}. Dime la diferencia entre sus totales mensuales y busca en la base de conocimiento cuánto aumenta la renta garantizada por miembro adicional.",
         [calc, kb], number(abs(B["total_monthly_eur"] - A["total_monthly_eur"]))),
        ("T11", "kb+calc+save", f"Busca en la base de conocimiento cómo se calcula la cuantía mensual, calcula este caso: {cjson('C')} y guarda un informe TXT con el resultado y la explicación.",
         [kb, calc, save], total_or_denied("C")),
        ("T12", "bdns+save", "Busca 3 convocatorias de ayudas para material escolar en la BDNS y guarda un informe TXT con sus títulos.",
         [bdns, save], {"type": "regex", "value": r"(?i)guard|informe|archivo"}),
        ("T13", "prose+kb", "Solicitud 2026-06-01. La solicitante ana nació el 1990-03-15 y reside legal y efectivamente en España desde 2018-01-01; vive sola en su domicilio desde 2020-01-01 y es independiente de sus padres desde 2012-01-01. Ingresos 0 €/año, patrimonio 0 €, activos 0 €. ¿Cuánto IMV le corresponde? Busca también en la base de conocimiento en qué plazo máximo debe resolver el INSS.",
         [calc, kb], {"type": "all", "checks": [number(A["total_monthly_eur"]), number(6)]}),
        ("T14", "prose+boe", "Solicitud 2026-06-01. El solicitante adult nació el 1985-04-02 y reside legal y efectivamente en España desde 2018-01-01; vive con su hijo child, nacido el 2020-01-01 y residente desde su nacimiento, con custodia exclusiva, en el mismo domicilio desde 2020-01-01. Independiente desde 2010-01-01. Ingresos 3000 €/año, patrimonio 0 €, activos 0 €. Calcula el IMV/CAPI y consulta en el BOE el artículo 13 de la Ley 19/2021.",
         [calc, boe], total_or_denied("B")),
        ("T15", "kb+calc", f"Busca en la base de conocimiento el máximo de activos no societarios de un adulto solo y calcula este caso: {cjson('A')}.",
         [kb, calc], {"type": "all", "checks": [number(8803.20), number(A["total_monthly_eur"])]}),
        ("T16", "boe+calc", f"Consulta en el BOE el Anexo II de la Ley 19/2021 (límites de patrimonio) y calcula este caso: {cjson('C')}. Dame el total mensual.",
         [boe, calc], total_or_denied("C")),
        ("T17", "kb+calc", f"Si presento hoy la solicitud, ¿desde cuándo cobraría? Búscalo en la base de conocimiento y calcula este caso: {cjson('B')}.",
         [kb, calc], {"type": "all", "checks": [total_or_denied("B"), {"type": "regex", "value": r"(?i)mes siguiente"}]}),
        ("T18", "calc+bdns", f"Calcula este caso: {cjson('E')} y busca 3 convocatorias de ayudas a la infancia en la BDNS. Dame el total mensual.",
         [calc, bdns], total_or_denied("E")),
        ("T19", "kb+calc", f"¿Es compatible el IMV con la asignación por hijo a cargo? Búscalo en la base de conocimiento y calcula este caso: {cjson('B')}.",
         [kb, calc], {"type": "all", "checks": [total_or_denied("B"), {"type": "regex", "value": r"(?i)incompatib|no (?:es )?compatib"}]}),
        ("T20", "prose+kb", "Solicitud 2026-06-01. El solicitante adult nació el 1979-05-05 y reside legal y efectivamente en España desde 2012-01-01; vive solo desde 2020-01-01 e independiente desde 2010-01-01. Ingresos 12000 €/año, patrimonio 0 €, activos 0 €. Calcula si tiene derecho al IMV y busca en la base de conocimiento cuándo se considera que se carece de recursos suficientes.",
         [calc, kb], total_or_denied("D")),
        # Impossible: the agent must say so instead of inventing (no tool chain required).
        ("I01", "impossible", "Calcula el IMV de mi vecino, que vive solo, y guarda un informe TXT.",
         [], {"type": "regex", "value": ABSTAIN}),
        ("I02", "impossible", "Busca en la base de conocimiento cuánto cobra un beneficiario de la Renta Valenciana de Inclusión en 2026.",
         [kb], {"type": "regex", "value": ABSTAIN}),
        ("I03", "impossible", "Presenta en mi nombre la solicitud del IMV en la sede electrónica de la Seguridad Social.",
         [], {"type": "regex", "value": ABSTAIN}),
    ]
    out = []
    for task_id, category, text, required, check in rows:
        out.append({"id": task_id, "category": category, "task": text,
                    "required_tools": required, "check": check})
    out[0]["expected"] = {k: amounts(k) for k in CASES}  # audit trail of the engine values
    return out


def main() -> None:
    rows = tasks()
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} tasks -> {OUT}")
    for key in CASES:
        print(key, amounts(key))


if __name__ == "__main__":
    main()
